import pika
import sys, os
import argparse
from absl import logging
import fire
import contextlib
from time import sleep
import logging
from . import eval_version, mq_channel, parse_evaluation_result, QUEUE_EXPIRE
import json
from datetime import datetime
import subprocess


def get_code_revision():
    try:
        # Get the remote URL of the current branch
        # Assumes 'origin' as the remote name; adjust accordingly if different
        command = "git rev-parse HEAD"
        result = subprocess.run(
            command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )

        # Decode bytes to string (stdout is a byte string)
        url = result.stdout.decode().strip()

        return url
    except subprocess.CalledProcessError as e:
        # Handle errors from subprocess
        print(f"Error occurred: {e}")
        return None


class MQEvaluationClient:
    def __init__(self, model_dir, exp_id, revision):
        logging.getLogger("pika").setLevel(logging.WARNING)
        if "EXP_DOMAIN" not in os.environ:
            raise ValueError("Please set environment variable EXP_DOMAIN")
        self.username = os.environ.get("_USER", os.environ["LOGNAME"])
        self.exchange_name = os.environ["EXP_DOMAIN"]
        self.evaluator_id = (
            f"{self.username}.{self.exchange_name}.evaluator_{eval_version}"
        )
        self.reply_routing_key = f"{self.username}.{self.exchange_name}.{exp_id}"
        self.channel = None
        self.last_eval = None
        self.model_dir = model_dir
        self.exp_id = exp_id
        if revision == "":
            revision = get_code_revision()
        else:
            self.revision = revision
        with mq_channel() as channel:
            channel.exchange_declare(
                exchange=self.exchange_name, exchange_type="direct"
            )

            channel.queue_declare(
                queue=self.evaluator_id,
                durable=True,
                arguments={"x-expires": QUEUE_EXPIRE},
            )
            channel.queue_declare(
                queue=self.reply_routing_key,
                durable=True,
                arguments={"x-expires": QUEUE_EXPIRE},
            )

            channel.queue_bind(
                exchange=self.exchange_name,
                queue=self.evaluator_id,
                routing_key=self.evaluator_id,
            )
            channel.queue_bind(
                exchange=self.exchange_name,
                queue=self.reply_routing_key,
                routing_key=self.reply_routing_key,
            )

    def request_evaluation(self, eval_step):
        try:
            with mq_channel() as channel:
                channel.basic_publish(
                    exchange=self.exchange_name,
                    routing_key=self.evaluator_id,
                    body=json.dumps(
                        {
                            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "revision": self.revision,
                            "model_dir": self.model_dir,
                            "exp_id": self.exp_id,
                            "reply_routing_key": self.reply_routing_key,
                            "eval_step": eval_step,
                        }
                    ),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                    ),
                )
                self.last_eval = eval_step
        except Exception as e:
            print(e)

    def get_evaluation_result(self):
        try:
            with mq_channel() as channel:
                method_frame, properties, body = channel.basic_get(
                    queue=self.reply_routing_key, auto_ack=True
                )
            if method_frame:
                res = parse_evaluation_result(body)
                if hasattr(res, "command"):
                    return res
                if res.eval_step == self.last_eval:
                    self.last_eval = None
                if res.results is not None:
                    return res
                return None
            else:
                return None
        except Exception as e:
            print(e)
            return None

    def wait4clean_up(self):
        results = []
        while self.last_eval is not None:
            sleep(5)
            try:
                with mq_channel() as channel:
                    method_frame, properties, body = channel.basic_get(
                        queue=self.reply_routing_key, auto_ack=True
                    )
                if method_frame:
                    res = parse_evaluation_result(body)
                    if res.eval_step == self.last_eval:
                        self.last_eval = None
                    if res.results is not None:
                        results.append(res)
            except Exception as e:
                print(e)
        try:
            with mq_channel() as channel:
                channel.queue_delete(queue=self.reply_routing_key)
        except Exception as e:
            print(e)
        return results
