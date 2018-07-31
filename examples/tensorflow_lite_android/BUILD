package(default_visibility = ['//visibility:public'])
licenses(["notice"]) # MIT license

load('@gmaven_rules//:defs.bzl', 'gmaven_artifact')

android_binary(
  name = 'net.marulabo.doodle',
  custom_package = 'net.marulabo.doodle',
  manifest = 'AndroidManifest.xml',
  srcs = glob(['src/**/*.java']),
  assets = ['@doodle.tflite//file'],
  assets_dir = '',
  nocompress_extensions = ['.tflite'],
  resource_files = glob(['res/**']),
  tags = ['manual'],
  deps = [
    gmaven_artifact('com.android.support:appcompat-v7:aar:25.1.1'),
    '@org_tensorflow_tensorflow_lite//aar',
  ],
)

android_binary(
  name = 'net.marulabo.doodle.native',
  custom_package = 'net.marulabo.doodle',
  manifest = 'AndroidManifest.xml',
  srcs = glob(['src/**/*.java']),
  assets = ['@doodle.tflite//file'],
  assets_dir = '',
  nocompress_extensions = ['.tflite'],
  resource_files = glob(['res/**']),
  tags = ['manual'],
  deps = [
    gmaven_artifact('com.android.support:appcompat-v7:aar:25.1.1'),
    '@org_tensorflow//tensorflow/contrib/lite/java:tensorflowlite',
  ],
)
