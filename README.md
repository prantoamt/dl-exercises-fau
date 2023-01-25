### Table of Contents  
---------------------

- [Welcome](#welcome)
- [Install Pdm Python](#install-pdm-python)
- [Clone repo](#clone-repo)
- [Goto the base directory](#goto-the-base-directory)
- [Install dependencies](#install-dependencies)
- [Run test cases](#run-test-cases)
    + [Exercise 0](#Exercise-0)
    + [Exercise 1](#Exercise-1)


## Welcome
----------------------------
Hello! Welcome to the **Deep Learning Exercises** repository. This `readme.md` file will guide you to run the exercises inside this repository on your local machine. In order to run the exercises, please pursue the following steps. We assume you are using `unix based operating system` and you have `python 3.8` installed. We have used [pdm python](https://pdm.fming.dev/latest/) as package manager. We have created few custom commands using pdm python as well. Therefore, it is strongly recommended to install `pdm python` beforehand to make life easier. Installation guidelines can be found on pdm python's website.

## Install Pdm Python
---------------------------
Follow the instructions of [Pdm Python's](https://pdm.fming.dev/latest/) website to install it on your local machine.


## Clone repo
--------------------
Clone this git repository to your local machine. Open your terminal and run the command
```
git clone git@github.com:prantoamt/dl-exercises-fau.git
```

## Goto the base directory
------------------------------------
After cloning the repo, goto the django project's base directory by executing the following command: 
```
cd dl-exercises-fau
```

## Install dependencies
-----------------------------------------------------------------
Now, install all dependencies using pdm python.
```
pdm install
```

## Run test cases
-------------------
Run the test cases of each excercise by executing the following commands. <br/>
#### Exercise 0: 
```
pdm run test
```