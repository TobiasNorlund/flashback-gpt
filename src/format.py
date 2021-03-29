import logging
import re
import json
from typing import Dict, Callable, List


def format_header(forum_title: str, thread_title: str):
    return f"{forum_title}\n{thread_title}\n\n"


def format_post_item(item: Dict, indent=0) -> str:
    if item["type"] == "quote":
        post_str = ('\t'*indent) + f"Citat: {item['username'] or 'Okänd användare'}\n"
        post_str += "".join(('\t'*indent) + f"{format_post_item(quote_item, indent+1)}" for quote_item in item["post"])
    elif item["type"] == "text":
        post_str = ('\t'*indent) + f"{item['text']}\n"
    return post_str


def format_thread_post(flashback_post: Dict):
    """
    Formats a flashback post
    :param flashback_post:
    :return: A string representation of the post
    """
    post_str = f"{flashback_post['username'] or 'Okänd användare'}:\n"
    for item in flashback_post["post"]:
        post_str += format_post_item(item)
    post_str += "\n"
    return post_str


def format_thread(flashback_thread: Dict,
                  split_predicate: Callable[[str, str], bool] = lambda r, p: False,
                  allow_empty: bool = False) -> List[str]:
    """
    Formats a Flashback thread to multiple textual records
    :param flashback_thread:
    :param split_predicate: Function (record: str, post: str) -> bool returning True if record should be split before
                            post
    :param allow_empty:     Whether to allow records with zero posts (only header)
    :return:
    """
    record_header = format_header(flashback_thread['forumTitle'], flashback_thread['threadTitle'])
    records = []
    current_record = record_header
    for post in flashback_thread["posts"]:
        post_str = format_thread_post(post)
        if split_predicate(current_record, post_str):
            # If we added post_str now, the record would be too long. Close record (as long as current_record != just header)
            if current_record != record_header:
                records.append(current_record)

            # Create new record and verify we can add post_str to that
            current_record = record_header
            if split_predicate(current_record, post_str):
                # logging.warning(f"Post {post['postId']} is too long to fit in a single record, skipping...")
                continue

        current_record += post_str

    # Close record (as long as current_record != just header)
    if current_record != record_header or allow_empty:
        records.append(current_record)
    return records


def parse_post_item(post_str, indent=0):
    line, *rest = post_str.split("\n", maxsplit=1)
    rest = rest[0] if rest else ""
    if re.match(r"\t*Citat: (.+)", line):
        # Quote post item
        username = re.findall(r"Citat: (.+)", line)[0]
        quote_post_items = []
        while rest.startswith("\t" * (indent+1)):
            quote_post_item, rest = parse_post_item(rest, indent=indent+1)
            quote_post_items.append(quote_post_item)
        return {"type": "quote", "username": username, "post": quote_post_items}, rest
    else: 
        # Text post item
        # Remove tabs
        text = line[indent:]
        return {"type": "text", "text": text}, rest


def parse_thread_post(thread_post_str: str):
    username, *rest = thread_post_str.split(":\n", maxsplit=1)
    rest = rest[0] if rest else ""
    post = []

    while len(rest) > 0:
        # Parse next post item
        post_item, rest = parse_post_item(rest)
        post.append(post_item)

    thread_post = {
        "username": username,
        "post": post
    }

    return thread_post


def parse_thread(thread_str: str):
    """
    Parses a formatted thread string into a structured dict representation. Basically the inverse of
    `format_thread` function
    """
    try:
        forum_title, rest = thread_str.split("\n", maxsplit=1)
        thread_title, rest = rest.split("\n\n", maxsplit=1)

        posts = []
        while len(rest) > 0:
            # Parse thread post
            thread_post_str, *rest = rest.split("\n\n", maxsplit=1)
            rest = rest[0] if rest else ""
            thread_post = parse_thread_post(thread_post_str)
            posts.append(thread_post)

        thread = {
            "forumTitle": forum_title,
            "threadTitle": thread_title,
            "posts": posts
        }

        return thread

    except Exception as e:
        raise e
        

if __name__ == "__main__":
    import sys
    import json
    import math
    import argparse
    from .dataset import get_tokenizer

    tokenizer = get_tokenizer()

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-record-length", type=int, default=math.inf)
    parser.add_argument("--separator", default="------------------")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    def split_pred(r, p):
        return len(tokenizer.encode(r)) + len(tokenizer.encode(p)) > args.max_record_length
    
    for i, line in enumerate(sys.stdin):
        for j, record in enumerate(format_thread(json.loads(line), split_predicate=split_pred)):
            if args.json:
                print(json.dumps({"thread": record}, ensure_ascii=False))
            else:
                print(record)
                print(args.separator)
