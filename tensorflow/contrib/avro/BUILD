package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_custom_op_py_library",
    "tf_custom_op_library",
    "tf_gen_op_wrapper_py",
    "tf_gen_op_libs",
)

cc_library(
    name = "avro_utils_api",
    hdrs = ["utils/avro_value.h",
            "utils/avro_mem_reader.h",
            "utils/prefix_tree.h",
            "utils/value_buffer.h",
            "utils/avro_parser.h",
            "utils/avro_parser_tree.h",
            "utils/avro_reader.h",
            "utils/avro_attr_parser.h"],
    deps = [
        "//third_party/avro",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "avro_utils",
    srcs = ["utils/avro_mem_reader.cc",
            "utils/prefix_tree.cc",
            "utils/value_buffer.cc",
            "utils/avro_parser.cc",
            "utils/avro_parser_tree.cc",
            "utils/avro_reader.cc",
            "utils/avro_attr_parser.cc"],
    deps = [
        ":avro_utils_api",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:core_cpu",
    ],
)

tf_cc_test(
    name = "avro_mem_reader_test",
    srcs = ["utils/avro_mem_reader_test.cc"],
    deps = [
        ":avro_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "prefix_tree_test",
    srcs = ["utils/prefix_tree_test.cc"],
    deps = [
        ":avro_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "value_buffer_test",
    srcs = ["utils/value_buffer_test.cc"],
    deps = [
        ":avro_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "avro_parser_test",
    srcs = ["utils/avro_parser_test.cc"],
    deps = [
        ":avro_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "avro_parser_tree_test",
    srcs = ["utils/avro_parser_tree_test.cc"],
    deps = [
        ":avro_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_custom_op_py_library(
    name = "avro",
    srcs = ["python/avro_record_dataset.py",
            "python/parse_avro_record.py",
            "python/avro_dataset.py",
            "//tensorflow/contrib/avro/python/utils:avro_utils"],
    dso = [":python/_avro_record_dataset.so",
           ":python/_parse_avro_record.so",
           ":python/_avro_dataset.so",
           "//third_party/avro:libavro_so"
    ],
    kernels = [
        ":avro_record_dataset_op_lib",
        ":parse_avro_record_op_lib",
        ":avro_dataset_op_lib",
    ],
    deps = [
         ":avro_record_dataset",
         ":parse_avro_record",
         ":avro_dataset",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

py_library(
    name = "avro_test",
    srcs = ["python/tests/avro_dataset_test_base.py"],
    deps = [],
    srcs_version = "PY2AND3",
)

tf_custom_op_library(
    name = "python/_avro_record_dataset.so",
    srcs = ["ops/avro_record_dataset.cc"],
    deps = ["//third_party/avro"],
)

tf_custom_op_library(
    name = "python/_parse_avro_record.so",
    srcs = ["ops/parse_avro_record.cc"],
    deps = [
        "//third_party/avro",
        "@com_googlesource_code_re2//:re2",
    ],
)

tf_custom_op_library(
    name = "python/_avro_dataset.so",
    srcs = ["ops/avro_dataset.cc",
            "utils/avro_mem_reader.cc",
            "utils/prefix_tree.cc",
            "utils/value_buffer.cc",
            "utils/avro_parser.cc",
            "utils/avro_parser_tree.cc",
            "utils/avro_reader.cc",
            "utils/avro_attr_parser.cc"],
    deps = [
        ":avro_utils_api", # Note, can't use avro utils because of TF core dependency
    ],
)

tf_gen_op_libs(
    op_lib_names = ["avro_record_dataset"],
    deps = [
         "//third_party/avro"
    ],
)

tf_gen_op_libs(
    op_lib_names = ["parse_avro_record"],
    deps = [
         "//third_party/avro"
    ],
)

tf_gen_op_libs(
    op_lib_names = ["avro_dataset"],
    deps = [
        ":avro_utils",
        "//third_party/avro",
    ],
)

tf_gen_op_wrapper_py(
    name = "avro_record_dataset",
    deps = [
         ":avro_record_dataset_op_lib",
         "//third_party/avro"
    ],
)

tf_gen_op_wrapper_py(
    name = "parse_avro_record",
    deps = [
         ":parse_avro_record_op_lib",
         "//third_party/avro"
    ],
)

tf_gen_op_wrapper_py(
    name = "avro_dataset",
    deps = [
         ":avro_dataset_op_lib",
         "//third_party/avro"
    ],
)

py_test(
    name = "avro_record_dataset_test",
    size = "small",
    srcs = ["python/tests/avro_record_dataset_test.py"],
    deps = [
        ":avro",
        "//tensorflow/contrib/avro/python/utils:avro_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python/data/util:convert",
    ],
)

py_test(
    name = "parse_avro_record_test",
    size = "medium",
    srcs = ["python/tests/parse_avro_record_test.py"],
    deps = [
        ":avro",
        "//tensorflow/contrib/avro/python/utils:avro_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:parsing_ops",
    ],
)

py_test(
    name = "avro_dataset_test",
    size = "small",
    srcs = ["python/tests/avro_dataset_test.py"],
    deps = [
        ":avro",
        ":avro_test",
        "//tensorflow/contrib/avro/python/utils:avro_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:parsing_ops",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
