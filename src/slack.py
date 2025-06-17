import json

import requests

SLACK_WEB_HOOK_URL = (
    "https://hooks.slack.com/services/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"
)


def send_slack_message(message):
    """
    Send a message to Slack using the webhook URL

    Args:
        message (str): Message to send to Slack
    """
    payload = {"text": message}

    try:
        response = requests.post(
            SLACK_WEB_HOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send Slack message: {e}")
