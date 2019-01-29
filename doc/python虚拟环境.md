# virtualenv

本教程将引导您完成安装和使用 Python 包。

它将向您展示如何安装和使用必要的工具，并就最佳做法做出强烈推荐。请记住， Python 用于许多不同的目的。准确地说，您希望如何管理依赖项，这些依赖项可能会根据您如何决定发布软件而发生变化。这里提供的指导最直接适用于网络服务 （包括 Web 应用程序）的开发和部署，但也非常适合管理任意项目的开发和测试环境。



## 确保您已经有了 Python 和 pip

在您进一步之前，请确保您有 Python，并且可从您的命令行中获得。 您可以通过简单地运行以下命令来检查：

```
$ python --version
```

您应该得到像 `3.6.2` 之类的一些输出。如果没有 Python，请从 [python.org](https://python.org/) 安装相应的版本。



另外，您需要确保 [pip](https://pypi.org/project/pip/) 是可用的。您可以通过运行以下命令来检查：

```
$ pip --version
```

如果您使用 [python.org](https://python.org/) 的安装程序来安装 Python，您应该已经有 pip 了。 如果您使用的是Linux，并使用操作系统的包管理器进行安装，则可能需要单独 [安装 pip](https://pip.pypa.io/en/stable/installing/)。

## 安装virtualenv

在您进一步之前，请确保您有 Python，并且可从您的命令行中获得。 您可以通过简单地运行以下命令来检查：

[virtualenv](http://pypi.python.org/pypi/virtualenv) 是一个创建隔绝的Python环境的 工具。virtualenv创建一个包含所有必要的可执行文件的文件夹，用来使用Python工程所需的包。

通过pip安装virtualenv：

```
$ pip install virtualenv
```

测试您的安装：

```
$ virtualenv --version
```

## 基本使用

1. 为一个工程创建一个虚拟环境：

```
$ cd my_project_folder
$ virtualenv venv
```

`virtualenv venv` 将会在当前的目录中创建一个文件夹，包含了Python可执行文件， 以及 `pip` 库的一份拷贝，这样就能安装其他包了。虚拟环境的名字（此例中是 `venv` ） 可以是任意的；若省略名字将会把文件均放在当前目录。

在任何您运行命令的目录中，这会创建Python的拷贝，并将之放在叫做 `venv` 的文件中。

您可以选择使用一个Python解释器（比如``python2.7``）：

```
$ virtualenv -p /usr/bin/python2.7 venv
```

2. 要开始使用虚拟环境，其需要被激活：

```
$ source venv/bin/activate
```

当前虚拟环境的名字会显示在提示符左侧（比如说 `(venv)您的电脑:您的工程 用户名$）`以让您知道它是激活的。从现在起，任何您使用pip安装的包将会放在 `venv` 文件夹中， 与全局安装的Python隔绝开。

像平常一样安装包，比如：

```
$ pip install requests
```

1. 如果您在虚拟环境中暂时完成了工作，则可以停用它：

```
$ deactivate
```

这将会回到系统默认的Python解释器，包括已安装的库也会回到默认的。

要删除一个虚拟环境，只需删除它的文件夹。（要这么做请执行 `rm -rf venv` ）

然后一段时间后，您可能会有很多个虚拟环境散落在系统各处，您将有可能忘记它们的名字或者位置。

## 其他注意事项

运行带 `--no-site-packages` 选项的 `virtualenv` 将不会包括全局安装的包。 这可用于保持包列表干净，以防以后需要访问它。（这在 `virtualenv` 1.7及之后是默认行为）

为了保持您的环境的一致性，“冷冻住（freeze）”环境包当前的状态是个好主意。要这么做，请运行：

```
$ pip freeze > requirements.txt
```

这将会创建一个 `requirements.txt` 文件，其中包含了当前环境中所有包及 各自的版本的简单列表。您可以使用 `pip list` 在不产生requirements文件的情况下， 查看已安装包的列表。这将会使另一个不同的开发者（或者是您，如果您需要重新创建这样的环境） 在以后安装相同版本的相同包变得容易。

```
$ pip install -r requirements.txt
```

这能帮助确保安装、部署和开发者之间的一致性。

------

参考来源：[Pipenv & 虚拟环境](https://pythonguidecn.readthedocs.io/zh/latest/dev/virtualenvs.html)