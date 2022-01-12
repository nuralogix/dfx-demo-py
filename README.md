# `dfxdemo`

`dfxdemo` is a simple Python-based demo that demonstrates how to use the
DeepAffex™ Extraction SDK and DeepAffex™ Cloud API.

The demo can extract facial blood-flow from a video file or from a webcam, send
it to the DeepAffex™ Cloud for processing and display the results. (This
process is called 'making a measurement'.) It can also be used to display
historical results or to view study details.

## Register for an DeepAffex™ Cloud API license

If you haven't already done so, the first step is to register for a DeepAffex™
developer account and request a cloud API license key. You can do this by
visiting the [DFX Dashboard](https://dashboard.deepaffex.ai/) and selecting the
Sign Up link at the bottom.

## Copy your license key

Once you are logged in to the Dashboard, you will see a Licenses section on the
left. Clicking on it will reveal your organization license that you can use in
the subsequent steps. Please click on the eye-icon (👁️) to reveal your
license and copy and save it locally. (You will need it for registering later.)

## Prerequisites

Please ensure you have the following software installed:

* [Python 3.7 or newer (64-bit)](https://www.python.org/)
* [Git](https://git-scm.com/)
* C++ compiler and toolchain, capable of compiling Python extensions:
  * Windows: [Visual Studio 2017 or newer](https://visualstudio.microsoft.com/)
  * macOS (untested): [Xcode](https://developer.apple.com/xcode/)
  * Linux: gcc

  Note: On Ubuntu 18.04, the following commands should work

  ```shell
  sudo apt-get install build-essential git # Compiler and Git
  sudo apt-get install python3.8-dev python3.8-venv # or 3.7
  sudo apt-get install libopenblas-dev liblapack-dev # for Dlib
  ```

Note: Please see the section [Using Docker](#using-docker) for an alternative.

## Install `dfxdemo`

Clone the dfxdemo application from Github.

```shell
git clone https://github.com/nuralogix/dfx-demo-py
```

Create a Python virtual environment inside the cloned repo, activate it and
upgrade `pip`

  ```shell
  cd dfx-demo-py
  python3 -m venv venv
  source venv/bin/activate # on Windows: venv\Scripts\activate
  python -m pip install --upgrade pip setuptools
  python -m pip install --upgrade wheel cmake
  ```

Download the [Python wheel for the DFX SDK](https://deepaffex.ai/developers-sdk)
for your platform and install it in the Python virtual environment.

```shell
pip install /path/to/download/libdfxpython.whl
```

Install `dfxdemo` in editable mode (and automatically install other
dependencies.) This may take a while as [Dlib](http://dlib.net/) gets compiled.

```shell
pip install -e ".[dlib]"
```

## Run `dfxdemo`

`dfxdemo` has top-level commands that roughly correspond to the way the DFX API
is organized. All commands and subcommands have a `--help` argument.

### Register your license

Register your organization license on the DeepAffex™ Cloud to obtain a *device
token*. This is generally the first thing you have to do (unless you don't want
to make a measurement.)

```shell
dfxdemo org register <your_license_key>
```

**Note**: By default, the demo stores tokens in a file called `config.json`.
*In a production application, you will need to manage all tokens securely.*

### Login

Login as a user to obtain a *user token*.

```shell
dfxdemo user login <email> <password>
```

**Note**: All the commands below, use the tokens obtained above.

---

### Studies

List the available DFX Studies and retrieve the details of the one you want to
use.

Note: The DeepAffex™ Cloud organizes around the concept of Studies - a DFX
Study is a collection of biosignals of interest that are computed in one
measurement.

```shell
dfxdemo studies list
dfxdemo study get <study_id>
```

Select a study for use in measurements.

```shell
dfxdemo study select <study_id>
```

### Measurements

Make a measurement from a video using the selected study

```shell
dfxdemo measure make /path/to/video_file
```

or

Make a measurment from a webcam using the selected study

```shell
dfxdemo measure make_camera
```

Retrieve detailed results of the last measurement

```shell
dfxdemo measure get
```

List historical measurements

```shell
dfxdemo measurements list
```

## Additional instructions for developers in China

If you intend to use the DeepAffex™ Cloud API in mainland China, please save
the following snippet as `config.json` in the same folder as `dfxdemo.py`
*before* using the example.

## 中国大陆的开发者请注意

DeepAffex™云端API在中国大陆使用一个不同于默认地址的URL。请将以下json代码在dfxdemo.py的同一目录下保存为config.json再使用。

```json
{
    "rest_url": "https://api.deepaffex.cn",
    "ws_url": "wss://api.deepaffex.cn"
}
```

## Using Docker

You can experiment with `dfxdemo` using Docker by following the instructions
below. There are a few limitations:

* `measure make_camera` will not work since the container doesn't have access
  to a camera
* `--headless` needs to be passed to `measure make` since the container doesn't
  have access to a X-server.

### Build the image

```shell
docker build . -t dfxdemo --build-arg EXTRAS_REQUIRE=dlib
docker image prune -f  # Optional
```

### Run the `dfxdemo` container

In the commands below, please replace `${PWD}` with `%CD%` on Windows.

```shell
# To run most commands, use this, assuming ${PWD} contains config.json etc.
docker run -it --rm -v ${PWD}:/app dfxdemo org register <your_license_key>

# To run `measure make`, use this, updating /path/to/videos to a path on your machine...
docker run -it --rm -v ${PWD}:/app -v /path/to/videos:/videos dfxdemo measure make /videos/video_file
```

### Other options

If you don't need to make measurements, then you can build without a facetracker.

```shell
docker build . -t dfxdemo --build-arg EXTRAS_REQUIRE=dummy
```

## Additional resources

* [Developers Guide](http://docs.deepaffex.ai/guide/index.html)
* [SDK](https://deepaffex.ai/developers-sdk)
  * [C++ Docs](http://docs.deepaffex.ai/c/index.html)
  * [Python Docs](http://docs.deepaffex.ai/python/index.html) **TODO**
  * [.NET Docs](http://docs.deepaffex.ai/dotnet/index.html)
* [Cloud API](https://deepaffex.ai/developers-api)
  * [Apiary](https://dfxapiversion10.docs.apiary.io/)
