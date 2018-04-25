
# Example Web Application

This application can test the model on the web.
The `/api/prediction` endpoint proxies to a specific TensorFlow Serving API.
Therefore, the model TensorFlow Serving Server needs to be running somewhere.

![I drew a cat.](img/mycat.png)

## Usage

Resolve dependencies.

```shell
$ # python 2
$ pip install -r ./requirements.2.txt
```

```shell
$ # python 3
$ pip install -r ./requirements.3.txt
```

Running server.

```shell
$ python ./app.py
```

### Options

- `-h`, `--host`: server listen host.
- `-p`, `--port`: server listen port.
- `--tf-server-host`: TensorFlow Serving server host.
- `--tf-server-port`: TensorFlow Serving server port.

### Endpoints

- `/`: Web Application
- `/api/prediction`: Prediction. In/Out format is json.
