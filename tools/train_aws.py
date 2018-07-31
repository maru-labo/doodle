# coding: utf-8
# Copyright (c) 2018 一般社団法人 MaruLabo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os, logging, json
from uuid import uuid4

def get_logger():
  logger = logging.getLogger('doodle')
  logger.setLevel(logging.DEBUG)
  return logger
logger = get_logger()

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow

def train(source_dir, data_path='doodle/data', training_steps=20000, evaluation_steps=2000,
      train_instance_type='local', train_instance_count=1, run_tensorboard_locally=True,
      uid=None, role=None, bucket=None, profile_name=None):
  assert os.path.exists(source_dir)
  boto_session = boto3.Session(profile_name=profile_name)
  session = sagemaker.Session(boto_session=boto_session)
  role   = role   if role   is not None else sagemaker.get_execution_role()
  bucket = bucket if bucket is not None else session.default_bucket()
  uid  = uid  if uid  is not None else uuid4()
  logger.debug(session.get_caller_identity_arn())
  role = session.expand_role(role)

  params = {
    'train_tfrecord_file': 'train.tfr',
    'test_tfrecord_file' : 'test.tfr',
    'samples_per_epoch'  : 700000,
    'save_summary_steps' : 100,
  }

  output_path   = 's3://{}/doodle/model/{}/export'.format(bucket, uid)
  checkpoint_path = 's3://{}/doodle/model/{}/ckpt'  .format(bucket, uid)
  code_location   = 's3://{}/doodle/model/{}/source'.format(bucket, uid)
  base_job_name   = 'doodle-training-job-{}'.format(uid)
  data_dir    = 's3://{}/{}'.format(bucket, data_path)

  logger.info('uid          : {}'.format(uid))
  logger.info('execution_role     : {}'.format(role))
  logger.info('data_dir       : {}'.format(data_dir))
  logger.info('output_path      : {}'.format(output_path))
  logger.info('checkpoint_path    : {}'.format(checkpoint_path))
  logger.info('code_location    : {}'.format(code_location))
  logger.info('base_job_name    : {}'.format(base_job_name))
  logger.info('training_steps     : {}'.format(training_steps))
  logger.info('evaluation_steps   : {}'.format(evaluation_steps))
  logger.info('train_instance_count : {}'.format(train_instance_count))
  logger.info('train_instance_type  : {}'.format(train_instance_type))
  logger.info('hyperparameters    : {}'.format(json.dumps(params)))

  estimator = TensorFlow(
    hyperparameters=params,

    output_path=output_path,
    checkpoint_path=checkpoint_path,
    code_location=code_location,
    base_job_name=base_job_name,

    source_dir=source_dir,
    entry_point='doodle.py',
    framework_version='1.6',

    role=role,
    training_steps=training_steps,
    evaluation_steps=evaluation_steps,
    train_instance_count=train_instance_count,
    train_instance_type=train_instance_type)

  estimator.fit(data_dir, run_tensorboard_locally=run_tensorboard_locally)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--source-dir')
  parser.add_argument('-d', '--data-path', type=str, default='doodle/data')
  parser.add_argument('-p', '--profile', type=str, default=None)
  parser.add_argument('--instance-count', type=int, default=1)
  parser.add_argument('--instance-type', type=str, default='local')
  parser.add_argument('--training-steps', type=int, default=20000)
  parser.add_argument('--evaluation-steps', type=int, default=2000)
  parser.add_argument('--run-tensorboard-locally', type=bool, default=True)
  parser.add_argument('--uid', type=str, default=None)
  parser.add_argument('--role', type=str, default=None)
  parser.add_argument('--bucket', type=str, default=None)
  args = parser.parse_args()
  train(source_dir=args.source_dir, data_path=args.data_path,
      training_steps=args.training_steps,
      evaluation_steps=args.evaluation_steps,
      train_instance_type=args.instance_type,
      train_instance_count=args.instance_count,
      run_tensorboard_locally=args.run_tensorboard_locally,
      uid=args.uid, role=args.role, bucket=args.bucket,
      profile_name=args.profile)
