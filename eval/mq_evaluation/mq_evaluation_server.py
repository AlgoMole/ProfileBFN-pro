import pika
import sys, os
import argparse
import logging
import fire
import contextlib
from time import sleep
from eval.mq_evaluation import (
    eval_version,
    mq_channel,
    parse_evaluation_request,
    mq_eval_result_reply,
    QUEUE_EXPIRE,
)
from eval.exp import get_contact
import json
import os
import torch
import copy
import subprocess
import traceback


def process_fn(cmd_str, gpus="0"):
    logging.info(f"process_fn: {cmd_str} with gpus: {gpus}")
    _env = copy.copy(os.environ)
    _env.update({"CUDA_VISIBLE_DEVICES": gpus})
    subprocess.call(cmd_str, shell=True, env=_env)


def get_git_repo_url():
    try:
        # Get the remote URL of the current branch
        # Assumes 'origin' as the remote name; adjust accordingly if different
        command = "git config --get remote.origin.url"
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


def read_eval_revision(model_dir):
    try:
        with open(f"{model_dir}/eval_revision.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def main():
    if "EXP_DOMAIN" not in os.environ:
        raise ValueError("Please set environment variable EXP_DOMAIN")
    username = os.environ.get("_USER", os.environ["LOGNAME"])
    exchange_name = os.environ["EXP_DOMAIN"]
    evaluator_id = f"{username}.{exchange_name}.evaluator_{eval_version}"
    logging.warning(f"waiting on mq: {evaluator_id}")
    with mq_channel() as channel:
        channel.exchange_declare(exchange=exchange_name, exchange_type="direct")

        channel.queue_declare(
            queue=evaluator_id,
            durable=True,
            arguments={"x-expires": QUEUE_EXPIRE},
        )
        channel.queue_bind(
            exchange=exchange_name, queue=evaluator_id, routing_key=evaluator_id
        )
    while True:
        sleep(60)
        try:
            with mq_channel() as channel:
                method_frame, properties, body = channel.basic_get(
                    queue=evaluator_id, auto_ack=True
                )
            if method_frame:
                logging.warning(
                    f"{evaluator_id} get evaluation request: {body.decode('utf-8')}"
                )
                args = parse_evaluation_request(body)
                commit = read_eval_revision(args.model_dir) or args.revision
                if commit != args.revision:
                    logging.warning(
                        f"Revision {args.revision} is mapped to evaluation commit: {commit}"
                    )
                repo = get_git_repo_url()
                dirpath = f"/tmp/{args.exp_id}_{args.eval_step}_{commit}"
                process_fn(
                    " ".join(
                        [
                            f"mkdir -p {dirpath};",
                            f"cd {dirpath}; git init;",
                            f"git remote add origin {repo};",
                            f"git fetch --depth 1 origin {commit};",
                            f"git checkout FETCH_HEAD;"
                            f"python ./eval/exp/get_contact.py",
                            f"--model_dir {args.model_dir}",
                            f"--exp_id {args.exp_id} --eval_step {args.eval_step}",
                            f"--exchange_name {exchange_name}",
                            f"--reply_routing_key {args.reply_routing_key};",
                            f"cd / && rm -r {dirpath};",
                        ]
                    )
                )
        except Exception as e:
            logging.warning(traceback.format_exc())


if __name__ == "__main__":
    logging.getLogger("pika").setLevel(logging.WARNING)
    fire.Fire(main)
