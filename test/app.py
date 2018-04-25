import os, logging, traceback

import numpy as np
import tensorflow as tf

from grpc.beta.implementations import insecure_channel

from sagemaker.tensorflow.tensorflow_serving.apis.predict_pb2 import PredictRequest
from sagemaker.tensorflow.tensorflow_serving.apis.prediction_service_pb2 import beta_create_PredictionService_stub

import falcon

class InvalidImageError(Exception):
    pass

def create_request(num, images):
    images = np.asarray(images).astype(np.float32).flatten()
    image_proto = tf.make_tensor_proto(
        values=images, shape=[num, 28, 28, 1], dtype=tf.float32)
    request = PredictRequest()
    request.model_spec.name = 'doodle'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['image'].CopyFrom(image_proto)
    return request

class AppPage(object):
    def __init__(self, filename, hot=False):
        self.filename = filename
        self.hot = hot
        with open(filename, 'r') as f:
            self.content = f.read()
    
    def on_get(self, req, res):
        res.content_type = 'text/html'
        if self.hot:
            with open(self.filename, 'r') as f:
                res.body = f.read()
                return
        res.body = self.content

class Prediction(object):
    def __init__(self, host, port):
        self.logger = logging.getLogger('doodle.predictions')
        self.channel = insecure_channel(host, port)
        self.stub = beta_create_PredictionService_stub(self.channel)
        self.predict_keys = ['probabilities', 'classes']

    def on_post(self, req, res):
        self.logger.debug('aaaa')
        self.logger.debug(req.media)
        image = req.media.get('image', None)
        if image is None:
            raise falcon.HTTPMissingParam('image')
        if not isinstance(image, list):
            raise falcon.HTTPInvalidParam('This params is must be "list" type.', 'image')
        try:
            request = create_request(1, image)
            future  = self.stub.Predict.future(request, 5.0)
            predict = future.result().outputs
            res.media = {key: tf.make_ndarray(predict[key]).tolist()
                         for key in self.predict_keys}
        except InvalidImageError as e:
            raise falcon.HTTPBadRequest('Invalid image data.')
        except Exception as e:
            self.logger.error(e)
            traceback.print_exc()
            raise falcon.HTTPInternalServerError()

def get_api(tf_server_host, tf_server_port):
    root = os.path.dirname(os.path.abspath(__file__))
    app = falcon.API()
    app.add_route('/', AppPage(os.path.join(root,'res/index.html')))
    app.add_route('/api/prediction', Prediction(tf_server_host, tf_server_port))
    return app

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--host', default='127.0.0.1')
    parser.add_argument('-p', '--port', type=int, default=8080)
    parser.add_argument('--tf-server-host', default='127.0.0.1')
    parser.add_argument('--tf-server-port', type=int, default=8500)
    args = parser.parse_args()

    app = get_api(args.tf_server_host, args.tf_server_port)

    from wsgiref import simple_server
    httpd = simple_server.make_server(args.host, args.port, app)
    httpd.serve_forever()
