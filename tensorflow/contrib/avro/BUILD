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

#load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
#load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
#load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
#load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")

cc_library(
    name = "cc_utils",
    srcs = ["utils/avro_mem_reader.cc",
            "utils/prefix_tree.cc",
            "utils/avro_parser.cc"],
    hdrs = ["utils/avro_value.h",
            "utils/avro_mem_reader.h",
            "utils/prefix_tree.h",
            "utils/value_buffer.h",
            "utils/avro_parser.h"],
    deps = [
        "//third_party/avro",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:core_cpu",
        "@com_googlesource_code_re2//:re2",
    ],
)

tf_cc_test(
    name = "avro_mem_reader_test",
    srcs = ["utils/avro_mem_reader_test.cc"],
    deps = [
        ":cc_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "prefix_tree_test",
    srcs = ["utils/prefix_tree_test.cc"],
    deps = [
        ":cc_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "value_buffer_test",
    srcs = ["utils/value_buffer_test.cc"],
    deps = [
        ":cc_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "avro_parser_test",
    srcs = ["utils/avro_parser_test.cc"],
    deps = [
        ":cc_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_custom_op_py_library(
    name = "avro",
    srcs = ["python/avro_record_dataset.py",
            "python/parse_avro_record.py",
            "//tensorflow/contrib/avro/python/utils:avro_utils"],
    dso = [":python/_avro_record_dataset.so",
           ":python/_parse_avro_record.so",
           "//third_party/avro:libavro_so"
    ],
    kernels = [
        ":avro_record_dataset_op_lib",
        ":parse_avro_record_op_lib",
    ],
    deps = [
         ":avro_record_dataset",
         ":parse_avro_record",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
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

tf_gen_op_libs(
    op_lib_names = ["avro_record_dataset"],
    deps = [
         "//third_party/avro"
    ],
)

tf_gen_op_wrapper_py(
    name = "avro_record_dataset",
    deps = [
         ":avro_record_dataset_op_lib",
         "//third_party/avro"
    ],
)

tf_gen_op_libs(
    op_lib_names = ["parse_avro_record"],
    deps = [
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
    size = "small",
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
