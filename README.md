# `dfxdemo`

`dfxdemo` is a simple Python-based demo that demonstrates how to use the
DeepAffex SDK and DeepAffex Cloud API.

The demo can extract facial blood-flow from a video file, send it to the
DeepAffex Cloud for processing and display the results. (This process is called
'making a measurement'.) It can also be used to display historical results or
to view study details.

## Register for an DeepAffex Cloud API license

If you haven't already done so, the first step is to register for a DeepAffex
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
  * [Visual Studio 2017 or newer](https://visualstudio.microsoft.com/) on Windows
  * [Xcode](https://developer.apple.com/xcode/) on Apple (untested)
  * gcc etc. on Linux
    * On Ubuntu: `sudo apt-get install build-essential libopenblas-dev liblapack-dev`

## Install `dfxdemo`

1. Clone the dfxdemo application from Github.

    ```shell
    git clone https://github.com/nuralogix/dfx-demo-py
    ```

2. Create a Python virtual environment inside the cloned repo, activate it and
upgrade `pip`

    ```shell
    cd dfx-demo-py
    python3 -m venv venv
    source venv/bin/activate # on Windows: venv\Scripts\activate
    python -m pip install --upgrade pip
    ```

3. Download the [Python wheel for the DFX
SDK](https://deepaffex.ai/developers-sdk) for your platform and install it in
the Python virtual environment.

    ```shell
    pip install /path/to/download/libdfxpython.whl
    ```

4. Install `dfxdemo` in editable mode (and automatically install other
dependencies.) This may take a while as [Dlib](http://dlib.net/) gets compiled.

    ```shell
    pip install -e .
    ```

## Run `dfxdemo`

`dfxdemo` has top-level commands that roughly correspond to the way the DFX API
is organized . All commands have a `--help` argument you can pass to get more
details.

### Register your license

Register your organization license on the DeepAffex Cloud to obtain a device
token. This is generally the first thing you have to do (unless you don't want
to make a measurement.)

```shell
python dfxdemo.py orgs register <your_license_key>
```

**Note**: By default, the demo stores tokens in a file called `config.json`.
*In a production application, you will need to manage all tokens securely.*

### Login

Login as a user to obtain a user token.

```shell
python dfxdemo.py users login <email> <password>
```

---

**Note**: All the commands below, use the tokens obtained above.

### Studies

List the available DFX Studies and retrieve the details of the one you want to
use.

Note: The DeepAffex Cloud organizes around the concept of Studies - a DFX
Study is a collection of biosignals of interest that are computed in one pass.

```shell
python dfxdemo.py studies list
python dfxdemo.py studies get <study_id>
```

Select a study for use in measurements.

```shell
python dfxdemo.py studies select <study_id>
```

### Measurements

Make a measurement using the selected study

   ```shell
   python dfxdemo.py measure make /path/to/video_file
   ```

Retrieve detailed results of the last measurement

```shell
python dfxdemo.py measure get
```

List historical measurements

```shell
python dfxdemo.py measurements list
```

## Additional resources

* [DFX Developers Guide](http://docs.deepaffex.ai/developers/index.html) **TODO**
* [DFX SDK](https://deepaffex.ai/developers-sdk)
  * [C++ Docs](http://docs.deepaffex.ai/c/index.html)
  * [Python Docs](http://docs.deepaffex.ai/python/index.html) **TODO**
  * [.NET Docs](http://docs.deepaffex.ai/dotnet/index.html)
* [DFX Cloud API](https://deepaffex.ai/developers-api)
  * [Apiary](https://dfxapiversion10.docs.apiary.io/)
* [Description of DFX workflow](https://github.com/nuralogix/dfx-e2e-demo-python/blob/master/dfxexample.md)
