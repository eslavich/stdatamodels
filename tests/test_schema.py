import functools

import pytest

from stdatamodels import schema as st_schema


def test_find_fits_keyword():
    schema = {
        "type": "object",
        "properties": {
            "foo": {
                "type": "string",
                "fits_keyword": "PHOOEY",
            },
            "bar": {
                "type": "object",
                "properties": {
                    "baz": {
                        "type": "string",
                        "fits_keyword": "PHOOEY",
                    },
                },
            },
        },
    }
    # Test presence of keyword at multiple levels of nesting:
    assert st_schema.find_fits_keyword(schema, "PHOOEY") == ["foo", "bar.baz"]
    # Missing keyword:
    assert st_schema.find_fits_keyword(schema, "NOPE") == []

    schema = {
        "type": "object",
        "properties": {
            "extra_fits": {
                "type": "object",
                "properties": {
                    "type": "string",
                    "fits_keyword": "PHOOEY",
                },
            },
        },
    }
    # Do not recurse into a top-level node called "extra_fits":
    assert st_schema.find_fits_keyword(schema, "PHOOEY") == []


def test_search_schema():
    # Test search by property name:
    schema = {
        "type": "object",
        "properties": {
            "foo": {
                "type": "string",
                "title": "Foo title",
                "description": "Foo description",
            },
        },
    }
    assert st_schema.search_schema(schema, "foo") == [
        ("foo", "Foo title\n\nFoo description"),
    ]

    # Test search by title:
    schema = {
        "type": "object",
        "properties": {
            "bar": {
                "type": "string",
                "title": "Bar title, which mentions 'Foo'",
                "description": "Bar description",
            },
        },
    }
    assert st_schema.search_schema(schema, "foo") == [
        ("bar", "Bar title, which mentions 'Foo'\n\nBar description"),
    ]

    # Test search by description:
    schema = {
        "type": "object",
        "properties": {
            "bar": {
                "type": "string",
                "title": "Bar title",
                "description": "Bar description, which mentions 'Foo'",
            },
        },
    }
    assert st_schema.search_schema(schema, "foo") == [
        ("bar", "Bar title\n\nBar description, which mentions 'Foo'"),
    ]

    # No results:
    schema = {}
    assert st_schema.search_schema(schema, "foo") == []

    # Multiple results:
    schema = {
        "type": "object",
        "properties": {
            "foo": {
                "type": "object",
                "properties": {
                    "bar": {
                        "title": "Bar title",
                        "description": "Bar description",
                    },
                    "baz": {
                        "title": "Baz title",
                        "description": "Baz description",
                    },
                },
            },
        },
    }
    assert st_schema.search_schema(schema, "foo") == [
        ("foo", ""),
        ("foo.bar", "Bar title\n\nBar description"),
        ("foo.baz", "Baz title\n\nBaz description"),
    ]

def test_search_schema_results_repr():
    # No results:
    assert repr(st_schema.search_schema({}, "foo")) == ""

    # Multiple results:
    schema = {
        "type": "object",
        "properties": {
            "foo": {
                "type": "object",
                "properties": {
                    "bar": {
                        "title": "Bar title",
                        "description": "Bar description",
                    },
                    "baz": {
                        "title": "Baz title",
                        "description": "Baz description",
                    },
                },
            },
        },
    }
    assert repr(st_schema.search_schema(schema, "foo")) == """
foo

foo.bar
    Bar title  Bar description
foo.baz
    Baz title  Baz description
""".strip()


def assert_walk_schema_results(schema, expected_results):
    __tracebackhide__ = True
    results = []
    ctx = object()

    def _callback(subschema, path):
        results.append((subschema, path))

    st_schema.walk_schema(schema, _callback)

    assert results == expected_results


def test_walk_schema_not_traversal():
    schema = {
        "not": {
            "type": "object",
            "properties": {
                "foo": {
                    "type": "string",
                },
            },
        },
    }
    assert_walk_schema_results(schema, [
        (schema, []),
        (schema["not"], ["not"]),
        (schema["not"]["properties"]["foo"], ["not", "properties", "foo"])
    ])


@pytest.mark.parametrize("combiner", ["allOf", "anyOf", "oneOf"])
def test_walk_schema_combiner_traversal(combiner):
    schema = {
        combiner: [
            {
                "type": "object",
                "properties": {
                    "foo": {
                        "type": "string",
                    },
                },
            },
            {
                "type": "object",
                "properties": {
                    "bar": {
                        "type": "string",
                    },
                },
            },
        ],
    }
    assert_walk_schema_results(schema, [
        (schema, []),
        (schema[combiner][0], [combiner, 0]),
        (schema[combiner][0]["properties"]["foo"], [combiner, 0, "properties", "foo"]),
        (schema[combiner][1], [combiner, 1]),
        (schema[combiner][1]["properties"]["bar"], [combiner, 1, "properties", "bar"]),
    ])


def test_walk_schema_object_traversal():
    schema = {
        "type": "object",
        "properties": {
            "foo": {
                "type": "string",
            },
            "bar": {
                "type": "object",
                "properties": {
                    "baz": {
                        "type": "string",
                    },
                },
            },
        },
    }
    assert_walk_schema_results(schema, [
        (schema, []),
        (schema["properties"]["foo"], ["properties", "foo"]),
        (schema["properties"]["bar"], ["properties", "bar"]),
        (schema["properties"]["bar"]["properties"]["baz"], ["properties", "bar", "properties", "baz"]),
    ])
