import json
from format import format_thread_post, parse_thread


def test_format_thread_post():
    quote_test = {
                  "username": "Användare1",
                  "post": [
                      {"type": "quote", "username": "usr",
                       "post": [
                           {"type": "text",
                            "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit."},
                           {"type": "text",
                            "text": "Vivamus porta ullamcorper felis id facilisis."}]},
                      {"type": "text",
                       "text": "Sed consectetur accumsan condimentum."}]
                  }

    expected = "Användare1:\n"
    expected += "Citat: usr\n"
    expected += "\tLorem ipsum dolor sit amet, consectetur adipiscing elit.\n"
    expected += "\tVivamus porta ullamcorper felis id facilisis.\n"
    expected += "Sed consectetur accumsan condimentum.\n"
    expected += "\n"

    assert format_thread_post(quote_test) == expected


def test_parse_thread():
    thread_str = """Forum > Subforum
Trådnamn

användare1:
Citat: användare2
\tCitat: användare3
\t\tDenna text är skriven av användare 3
\tDenna text är skriven av användare 2
\tDenna text också
Denna text är skriven av användare 1
användare 1 har skrivit detta också

användare2:
Jag skriver igen

"""

    expected = {
        "forumTitle": "Forum > Subforum",
        "threadTitle": "Trådnamn",
        "posts": [
            {
                "username": "användare1",
                "post": [
                    {"type": "quote", "username": "användare2", "post": [
                        {"type": "quote", "username": "användare3", "post": [
                            {"type": "text", "text": "Denna text är skriven av användare 3"}
                        ]},
                        {"type": "text", "text": "Denna text är skriven av användare 2"},
                        {"type": "text", "text": "Denna text också"}
                    ]},
                    {"type": "text", "text": "Denna text är skriven av användare 1"},
                    {"type": "text", "text": "användare 1 har skrivit detta också"},
                ]
            },
            {
                "username": "användare2",
                "post": [
                    {"type": "text", "text": "Jag skriver igen"}
                ]
            }
        ]
    }

    parsed = parse_thread(thread_str)
    print(json.dumps(parsed, indent=4, sort_keys=True))
    print(json.dumps(expected, indent=4, sort_keys=True))

    assert parsed == expected, "parse_thread should return the expected dict"
