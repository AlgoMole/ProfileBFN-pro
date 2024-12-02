import contextlib
import pika
import json
from types import SimpleNamespace

eval_version = "003"  # auto expire queues
QUEUE_EXPIRE = 24 * 60 * 60 * 1000  #  24h


@contextlib.contextmanager
def mq_channel():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host="mq.jjgong.cloud",
            port=10567,
            virtual_host="gongjj",
            credentials=pika.PlainCredentials("gongjj_evaluator", "algomolepass"),
        ),
    )
    channel = connection.channel()
    yield channel
    channel.close()
    connection.close()


def mq_eval_result_reply(exchange_name, routing_key, message):
    with mq_channel() as channel:
        channel.exchange_declare(exchange=exchange_name, exchange_type="direct")
        channel.queue_declare(
            queue=routing_key, durable=True, arguments={"x-expires": QUEUE_EXPIRE}
        )
        channel.queue_bind(
            exchange=exchange_name, queue=routing_key, routing_key=routing_key
        )
        channel.basic_publish(
            exchange=exchange_name,
            routing_key=routing_key,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            ),
        )


def parse_evaluation_request(body):
    args = json.loads(body.decode("utf-8"), object_hook=lambda d: SimpleNamespace(**d))
    return args


def parse_evaluation_result(body):
    res = json.loads(body.decode("utf-8"), object_hook=lambda d: SimpleNamespace(**d))
    return res
