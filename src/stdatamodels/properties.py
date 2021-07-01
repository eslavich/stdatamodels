# Licensed under a 3-clause BSD style license - see LICENSE.rst

import copy
import numpy as np
from collections.abc import Mapping, MutableMapping, MutableSequence
from astropy.io import fits

from astropy.utils.compat.misc import override__dir__

from asdf.tags.core import ndarray
from jsonschema import ValidationError

from . import util
from . import validate
from . import schema as mschema

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


__all__ = ['ObjectNode', 'ListNode']


def _is_struct_array(val):
    return (isinstance(val, (np.ndarray, fits.FITS_rec)) and
            val.dtype.names is not None and val.dtype.fields is not None)


def _is_struct_array_precursor(val):
    return isinstance(val, list) and isinstance(val[0], tuple)


def _is_struct_array_schema(schema):
    return (isinstance(schema['datatype'], list) and
            any('name' in t for t in schema['datatype']))


def _cast(val, schema):
    val = _unmake_node(val)
    if val is None:
        return None

    if 'datatype' in schema:
        # Handle lazy array
        if isinstance(val, ndarray.NDArrayType):
            val = val._make_array()

        if (_is_struct_array_schema(schema) and len(val) and
            (_is_struct_array_precursor(val) or _is_struct_array(val))):
            # we are dealing with a structured array. Because we may
            # modify schema (to add shape), we make a deep copy of the
            # schema here:
            schema = copy.deepcopy(schema)

            for t, v in zip(schema['datatype'], val[0]):
                if not isinstance(t, Mapping):
                    continue

                aval = np.asanyarray(v)
                shape = aval.shape
                val_ndim = len(shape)

                # make sure that if 'ndim' is specified for a field,
                # it matches the dimensionality of val's field:
                if 'ndim' in t and val_ndim != t['ndim']:
                    raise ValueError(
                        "Array has wrong number of dimensions. "
                        "Expected {}, got {}".format(t['ndim'], val_ndim)
                    )

                if 'max_ndim' in t and val_ndim > t['max_ndim']:
                    raise ValueError(
                        "Array has wrong number of dimensions. "
                        "Expected <= {}, got {}".format(t['max_ndim'], val_ndim)
                    )

                # if shape of a field's value is not specified in the schema,
                # add it to the schema based on the shape of the actual data:
                if 'shape' not in t:
                    t['shape'] = shape

        dtype = ndarray.asdf_datatype_to_numpy_dtype(schema['datatype'])
        val = util.gentle_asarray(val, dtype)

        if dtype.fields is not None:
            val = _as_fitsrec(val)

    if 'ndim' in schema and len(val.shape) != schema['ndim']:
        raise ValueError(
            "Array has wrong number of dimensions.  Expected {}, got {}"
            .format(schema['ndim'], len(val.shape)))

    if 'max_ndim' in schema and len(val.shape) > schema['max_ndim']:
        raise ValueError(
            "Array has wrong number of dimensions.  Expected <= {}, got {}"
            .format(schema['max_ndim'], len(val.shape)))

    if isinstance(val, np.generic) and np.isscalar(val):
        val = val.item()

    return val


def _as_fitsrec(val):
    """
    Convert a numpy record into a fits record if it is not one already
    """
    if isinstance(val, fits.FITS_rec):
        return val
    else:
        coldefs = fits.ColDefs(val)
        uint = any(c._pseudo_unsigned_ints for c in coldefs)
        fits_rec = fits.FITS_rec(val)
        fits_rec._coldefs = coldefs
        # FITS_rec needs to know if it should be operating in pseudo-unsigned-ints mode,
        # otherwise it won't properly convert integer columns with TZEROn before saving.
        fits_rec._uint = uint
        return fits_rec


def _get_schema_type(schema):
    """
    Create a list of types used by a schema and its subschemas when
    the subschemas are joined by combiners. Then return a type string
    if all the types are the same or 'mixed' if they differ
    """
    def callback(subschema, path, combiner, types, recurse):
        if 'type' in subschema:
            types.append(subschema['type'])

        has_combiner = ('anyOf' in subschema.keys() or
                        'allOf' in subschema.keys())
        return not has_combiner

    types = []
    mschema.walk_schema(schema, callback, types)

    schema_type = None
    for a_type in types:
        if schema_type is None:
            schema_type = a_type
        elif schema_type != a_type:
            schema_type = 'mixed'
            break

    return schema_type


def _make_default_array(attr, schema, ctx):
    dtype = schema.get('datatype')
    if dtype is not None:
        dtype = ndarray.asdf_datatype_to_numpy_dtype(dtype)
    ndim = schema.get('ndim', schema.get('max_ndim'))
    default = schema.get('default', None)
    primary_array_name = ctx.get_primary_array_name()

    if attr == primary_array_name:
        if ctx.shape is not None:
            shape = ctx.shape
        elif ndim is not None:
            shape = tuple([0] * ndim)
        else:
            shape = (0,)
    else:
        if dtype.names is not None:
            if ndim is None:
                shape = (0,)
            else:
                shape = tuple([0] * ndim)
            default = None
        else:
            has_primary_array_shape = False
            if primary_array_name is not None:
                primary_array = getattr(ctx, primary_array_name, None)
                has_primary_array_shape = primary_array is not None

            if has_primary_array_shape:
                if ndim is None:
                    shape = primary_array.shape
                else:
                    shape = primary_array.shape[-ndim:]
            elif ndim is None:
                shape = (0,)
            else:
                shape = tuple([0] * ndim)

    array = np.empty(shape, dtype=dtype)
    if default is not None:
        array[...] = default
    return array


def _make_default(attr, schema, ctx):
    if 'max_ndim' in schema or 'ndim' in schema or 'datatype' in schema:
        return _make_default_array(attr, schema, ctx)
    elif 'default' in schema:
        return schema['default']
    else:
        schema_type = _get_schema_type(schema)
        if schema_type == 'object':
            return {}
        elif schema_type == 'array':
            return []
        else:
            return None


def _make_node(parent, key, instance, schema):
    if isinstance(instance, dict):
        return ObjectNode(parent._path + [key], instance, schema, parent._ctx)
    elif isinstance(instance, list):
        return ListNode(parent._path + [key], instance, schema, parent._ctx)
    else:
        return instance


def _unmake_node(obj):
    if isinstance(obj, Node):
        return obj.instance
    return obj


def _get_schema_for_property(schema, attr):
    subschema = schema.get('properties', {}).get(attr, None)
    if subschema is not None:
        return subschema
    for combiner in ['allOf', 'anyOf']:
        for subschema in schema.get(combiner, []):
            subsubschema = _get_schema_for_property(subschema, attr)
            if subsubschema != {}:
                return subsubschema
    return {}


def _get_schema_for_index(schema, i):
    items = schema.get('items', {})
    if isinstance(items, list):
        if i >= len(items):
            return {}
        else:
            return items[i]
    else:
        return items


def _find_property(schema, attr):
    subschema = _get_schema_for_property(schema, attr)
    if subschema == {}:
        find = False
    else:
        find = 'default' in subschema
    return find


class Node():
    def __init__(self, path, instance, schema, ctx):
        self._path = path
        self._instance = instance
        self._schema = schema
        self._ctx = ctx

    def _validate(self):
        return validate.value_change(self._path, self._instance, self._schema, self._ctx)

    @property
    def instance(self):
        return self._instance


class ObjectNode(Node, MutableMapping):
    @override__dir__
    def __dir__(self):
        return list(self._schema.get('properties', {}).keys())

    def __eq__(self, other):
        if isinstance(other, ObjectNode):
            return self._instance == other._instance
        else:
            return self._instance == other

    def __getattr__(self, attr):
        # TODO: Figure out if this is necessary
        if attr.startswith('_'):
            raise AttributeError(f"No attribute '{attr}'")

        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError(f"Attribute '{attr}' has not been assigned and has no default in the schema")

    def __setattr__(self, attr, val):
        # TODO: Figure out if this is necessary
        if attr.startswith('_'):
            self.__dict__[attr] = val
        else:
            try:
                self.__setitem__(attr, val)
            except KeyError:
                raise AttributeError(f"Attribute '{attr}' cannot be set")

    def __delattr__(self, attr):
        if attr.startswith('_'):
            del self.__dict__[attr]
        else:
            try:
                self.__delitem__(attr)
            except KeyError:
                raise AttributeError(f"Attribute '{attr}' missing")

    def __iter__(self):
        return NodeIterator(self._instance)

    def __len__(self):
        return sum(1 for _ in NodeIterator(self._instance))

    def __getitem__(self, key):
        parts = key.split('.', 1)

        schema = _get_schema_for_property(self._schema, parts[0])
        try:
            value = self._instance[parts[0]]
        except KeyError:
            if schema == {}:
                raise KeyError(f"No key '{key}'")
            value = _make_default(parts[0], schema, self._ctx)
            if value is not None:
                self._instance[parts[0]] = value

        node = _make_node(self, parts[0], value, schema)

        if len(parts) > 1:
            return node.__getitem__(parts[-1])
        else:
            return node

    def __setitem__(self, key, value):
        parts = key.split('.', 1)

        if len(parts) > 1:
            self.__getitem__(parts[0]).__setitem__(parts[-1], value)
        else:
            schema = _get_schema_for_property(self._schema, key)
            if value is None:
                # Setting None is interpreted as a request to restore
                # the default value.
                value = _make_default(key, schema, self._ctx)
            value = _cast(value, schema)

            if self._ctx._validate_on_assignment:
                if validate.value_change(self._path + [key], value, schema, self._ctx):
                    self._instance[key] = value
            else:
                self._instance[key] = value

    def __delitem__(self, key):
        parts = key.split('.', 1)

        if len(parts) > 1:
            self.__getitem__(parts[0]).__delitem__(parts[-1])
        else:
            if self._ctx._validate_on_assignment:
                # Deleting an item might run us afoul of a required
                # validator, so we need to validate the whole instance.
                test_instance = self._instance.copy()
                del test_instance[key]
                if validate.value_change(self._path, test_instance, self._schema, self._ctx):
                    del self._instance[key]
            else:
                del self._instance[key]


def _slice_to_str(slice):
    """
    Convert a slice object to its string representation
    for use in a DataModel key.
    """
    parts = []
    if slice.start is not None:
        parts.append(str(slice.start))
    parts.append(':')

    if slice.stop is not None:
        parts.append(str(slice.stop))

    if slice.step is not None:
        parts.append(':')
        parts.append(str(slice.step))

    return ''.join(parts)


def _str_to_slice(string):
    """
    Convert a string representation of a slice to a
    slice object.
    """
    parts = string.split(':')
    if parts[0] != '':
        start = int(parts[0])
    else:
        start = None

    if parts[1] != '':
        end = int(parts[1])
    else:
        end = None

    if len(parts) > 2 and parts[2] != '':
        step = int(parts[2])
    else:
        step = None

    return slice(start, end, step)


def _handle_key(key):
    """
    Split the current level off of a multi-level key.
    """
    if isinstance(key, (int, slice)):
        return key, None
        else:
            parts = key.split('.', 1)
            if ':' in parts[0]:
                index = _str_to_slice(parts[0])
            else:
                index = int(parts[0])


class ListNode(Node, MutableSequence):
    def __cast(self, other):
        if isinstance(other, ListNode):
            return other._instance
        return other

    def __repr__(self):
        return repr(self._instance)

    def __eq__(self, other):
        return self._instance == self.__cast(other)

    def __ne__(self, other):
        return self._instance != self.__cast(other)

    def __len__(self):
        return len(self._instance)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            parts = [key]
            index = key
        else:
            parts = key.split('.', 1)
            if ':' in parts[0]:
                index = _str_to_slice(parts[0])
            else:
                index = int(parts[0])

        if len(parts) > 1:
            schema = _get_schema_for_index(self._schema, index)
            node = _make_node(self, index, self._instance[index], schema)
            return node.__getitem__(parts[-1])
        elif isinstance(index, int):
            schema = _get_schema_for_index(self._schema, index)
            return _make_node(self, index, self._instance[index], schema)
        else:
            indices = list(range(*index.indices(len(self._instance))))

            if isinstance(self._schema['items'], list):
                schema_items = [_get_schema_for_index(self._schema, i) for i in indices]
            else:
                schema_items = self._schema['items']

            schema = {'type': 'array', 'items': schema_items}

            return _make_node(self, _slice_to_str(index), self._instance[index], schema, self._ctx)

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice)):
            parts = [key]
            index = key
        else:
            parts = key.split('.', 1)
            if ':' in parts[0]:
                index = _str_to_slice(parts[0])
            else:
                index = int(parts[0])

        if len(parts) > 1:
            self.__getitem__(index).__setitem__(parts[-1], value)
        elif isinstance(index, int):
            schema = _get_schema_for_index(self._schema, index)
            value = _cast(value, schema)

            if self._ctx._validate_on_assignment:
                if validate.value_change(self._path + [index], value, schema, self._ctx):
                    self._instance[index] = value
            else:
                self._instance[index] = value
        else:
            if self._ctx._validate_on_assignment:
                indices = list(range(*index.indices(len(self._instance))))
                schemas = [_get_schema_for_index(self._schema, i) for i in indices]
                values = [_cast(value, s) for s in schemas]
                if all(validate.value_change(self._path + [i], v, s, self._ctx) for i, v, s in zip(indices, values, schemas)):
                    self._instance[index] = value
            else:
                self._instance[index] = value

    def __delitem__(self, key):
        if isinstance(key, (int, slice)):
            parts = [key]
            index = key
        else:
            parts = key.split('.', 1)
            if ':' in parts[0]:
                index = _str_to_slice(parts[0])
            else:
                index = int(parts[0])

        if len(parts) > 1:
            self.__getitem__(index).__delitem__(parts[-1])
        else:
            if self._ctx._validate_on_assignment:
                # Deleting an item changes the indexes of the remainder of the
                # list, so we need to validate the whole instance.
                test_instance = self._instance.copy()
                del test_instance[index]
                if validate.value_change(self._path, test_instance, self._schema, self._ctx):
                    del self._instance[index]
            else:
                del self._instance[index]

    def insert(self, i, item):
        schema = _get_schema_for_index(self._schema, i)
        item = _cast(item, schema)

        if self._ctx._validate_on_assignment:
            # Inserting an item shifts all subsequent items to new indexes,
            # so we need to validate the entire instance.
            test_instance = self._instance.copy()
            test_instance.insert(i, item)
            if validate.value_change(self._path, test_instance, self._schema, self._ctx):
                self._instance.insert(i, item)
        else:
            self._instance.insert(i, item)

    def sort(self, *args, **kwargs):
        if self._ctx._validate_on_assignment:
            test_instance = self._instance.copy()
            test_instance.sort(*args, **kwargs)
            if validate.value_change(self._path, test_instance, self._schema, self._ctx):
                self._instance.sort(*args, **kwargs)
        else:
            self._instance.sort(*args, **kwargs)

    def item(self, **kwargs):
        """
        Create an element for this array. Schema must use "list validation" for array
        items and item type must be object.

        Parameters
        ----------
        **kwargs
            Initial values for the new object.

        Returns
        -------
        ObjectNode
        """
        if not isinstance(self._schema.get('items'), dict):
            raise TypeError('Schema for this array uses tuple validation')

        if not self._schema['items'].get('type') == 'object':
            raise TypeError('Schema for this array specifies non-object elements')

        return _make_node(self, 0, kwargs, self._schema['items'])


class NodeIterator:
    """
    An iterator over node keys which flattens the hierachical structure.
    """
    def __init__(self, node):
        self.key_stack = []
        self.iter_stack = [self._get_iter(node)]

    def __iter__(self):
        return self

    def __next__(self):
        while self.iter_stack:
            try:
                key, val = next(self.iter_stack[-1])
            except StopIteration:
                self.iter_stack.pop()
                if self.iter_stack:
                    self.key_stack.pop()
                continue

            if isinstance(val, (dict, list)):
                self.key_stack.append(key)
                self._get_iter(val)
            else:
                return '.'.join(self.key_stack + [str(key)])

        raise StopIteration

    def _get_iter(self, val):
        if isinstance(val, dict):
            return ((str(k), v) for k, v in val.items())
        elif isinstance(val, list):
            return ((str(i), v) for i, v in enumerate(val))
        else:
            raise TypeError("Can't make iterator")


def put_value(path, value, tree):
    """
    Put a value at the given path into tree, replacing it if it is
    already present.

    Parameters
    ----------
    path : list of str or int
        The path to the element.

    value : any
        The value to place

    tree : JSON object tree
    """
    cursor = tree
    for i in range(len(path) - 1):
        part = path[i]
        if isinstance(part, int):
            while len(cursor) <= part:
                cursor.append({})
            cursor = cursor[part]
        else:
            if isinstance(path[i + 1], int) or path[i + 1] == 'items':
                cursor = cursor.setdefault(part, [])
            else:
                cursor = cursor.setdefault(part, {})

    if isinstance(path[-1], int):
        while len(cursor) <= path[-1]:
            cursor.append({})
    cursor[path[-1]] = value


def merge_tree(a, b):
    """
    Merge elements from tree `b` into tree `a`.
    """
    def recurse(a, b):
        if isinstance(b, dict):
            if not isinstance(a, dict):
                return copy.deepcopy(b)
            for key, val in b.items():
                a[key] = recurse(a.get(key), val)
            return a
        return copy.deepcopy(b)

    recurse(a, b)
    return a
