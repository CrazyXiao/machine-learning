## 概述

这里我们主要探讨 Python 2.7 和 Python 3 之间的潜在可能性和关键的程序化差异，因为Python 2.7 具有 Python 2 和 Python 3.0 之间的过渡版本，对许多库具有兼容性，对于程序员而言一直是非常流行的选择。然而，Python 2.7 被认为是一种遗留语言，且它的后续开发，现在更多的是 bug 修复，将在 2020 年完全停止。



## 主要差异

被视为 Python 的未来的Python 3虽然 和Python 2.7 和有许多类似的功能，但它们不应该被认为是完全可互换的，Python 3最初在设计之初就是不向后兼容Python 2的。但好在这些差异不是很大，同时为了简化这个转换过程，Python 3自带了一个叫做`2to3`的实用脚本，帮你将你的Python 2程序源文件自动转换到Python 3的形式。下面列举一些主要差异。

### Print

在Python 2里，`print`是一个语句。无论你想输出什么，只要将它们放在`print`关键字后边就可以。在Python 3里，`print()`是一个函数。就像其他的函数一样，`print()`需要你将想要输出的东西作为参数传给它。

| Notes | Python 2                      | Python 3                          |
| ----- | ----------------------------- | --------------------------------- |
| ①     | `print`                       | `print()`                         |
| ②     | `print 1`                     | `print(1)`                        |
| ③     | `print 1, 2`                  | `print(1, 2)`                     |
| ④     | `print 1, 2,`                 | `print(1, 2, end=' ')`            |
| ⑤     | `print >>sys.stderr, 1, 2, 3` | `print(1, 2, 3, file=sys.stderr)` |

1. 为输出一个空白行，需要调用不带参数的`print()`。
2. 为输出一个单独的值，需要将这这个值作为`print()`的一个参数就可以了。
3. 为输出使用一个空格分隔的两个值，用两个参数调用`print()`即可。
4. 这个例子有一些技巧。在Python 2里，如果你使用一个逗号(,)作为`print`语句的结尾，它将会用空格分隔输出的结果，然后在输出一个尾随的空格(trailing space)，而不输出回车(carriage return)。在Python 3里，通过把`end=' '`作为一个关键字参数传给`print()`可以实现同样的效果。参数`end`的默认值为`'\n'`，所以通过重新指定`end`参数的值，可以取消在末尾输出回车符。
5. 在Python 2里，你可以通过使用`>>pipe_name`语法，把输出重定向到一个管道，比如`sys.stderr`。在Python 3里，你可以通过将管道作为关键字参数`file`的值传递给`print()`来完成同样的功能。参数`file`的默认值为`std.stdout`，所以重新指定它的值将会使`print()`输出到一个另外一个管道。

### 除法&整型

Python3中/表示真除，%表示取余，//结果取整；Python2中带上小数点/表示真除，%表示取余，//结果取整。

Python 2有为非浮点数准备的`int`和`long`类型。`int`类型的最大值不能超过`sys.maxint`，而且这个最大值是平台相关的。可以通过在数字的末尾附上一个`L`来定义长整型，显然，它比`int`类型表示的数字范围更大。在Python 3里，只有一种整数类型`int`，大多数情况下，它很像Python 2里的长整型。由于已经不存在两种类型的整数，所以就没有必要使用特殊的语法去区别他们。

| Notes | Python 2              | Python 3             |
| ----- | --------------------- | -------------------- |
| ①     | `x = 1000000000000L`  | `x = 1000000000000`  |
| ②     | `x = 0xFFFFFFFFFFFFL` | `x = 0xFFFFFFFFFFFF` |
| ③     | `long(x)`             | `int(x)`             |
| ④     | `type(x) is long`     | `type(x) is int`     |
| ⑤     | `isinstance(x, long)` | `isinstance(x, int)` |

1. 在Python 2里的十进制长整型在Python 3里被替换为十进制的普通整数。
2. 在Python 2里的十六进制长整型在Python 3里被替换为十六进制的普通整数。
3. 在Python 3里，由于长整型已经不存在了，自然原来的`long()`函数也没有了。为了强制转换一个变量到整型，可以使用`int()`函数。
4. 检查一个变量是否是整型，获得它的数据类型，并与一个`int`类型(不是`long`)的作比较。
5. 你也可以使用`isinstance()`函数来检查数据类型；再强调一次，使用`int`，而不是`long`，来检查整数类型。

### Unicode字符串

Python 2有两种字符串类型：Unicode字符串和非Unicode字符串。Python 3只有一种类型：Unicode字符串。

| Notes | Python 2             | Python 3            |
| ----- | -------------------- | ------------------- |
| ①     | `u'PapayaWhip'`      | `'PapayaWhip'`      |
| ②     | `ur'PapayaWhip\foo'` | `r'PapayaWhip\foo'` |

1. Python 2里的Unicode字符串在Python 3里即普通字符串，因为在Python 3里字符串总是Unicode形式的。
2. Unicode原始字符串(raw string)(使用这种字符串，Python不会自动转义反斜线"\")也被替换为普通的字符串，因为在Python 3里，所有原始字符串都是以Unicode编码的。

Python 2有两个全局函数可以把对象强制转换成字符串：`unicode()`把对象转换成Unicode字符串，还有`str()`把对象转换为非Unicode字符串。Python 3只有一种字符串类型，Unicode字符串，所以`str()`函数即可完成所有的功能。(`unicode()`函数在Python 3里不再存在了。)

| Notes | Python 2            | Python 3        |
| ----- | ------------------- | --------------- |
|       | `unicode(anything)` | `str(anything)` |

### 全局函数`apply()`

Python 2有一个叫做`apply()`的全局函数，它使用一个函数f和一个列表`[a, b, c]`作为参数，返回值是`f(a, b, c)`。你也可以通过直接调用这个函数，在列表前添加一个星号(*)作为参数传递给它来完成同样的事情。在Python 3里，`apply()`函数不再存在了；必须使用星号标记法。

| Notes | Python 2                                                     | Python 3                                                    |
| ----- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| ①     | `apply(a_function, a_list_of_args)`                          | `a_function(*a_list_of_args)`                               |
| ②     | `apply(a_function, a_list_of_args, a_dictionary_of_named_args)` | `a_function(*a_list_of_args, **a_dictionary_of_named_args)` |
| ③     | `apply(a_function, a_list_of_args + z)`                      | `a_function(*a_list_of_args + z)`                           |
| ④     | `apply(aModule.a_function, a_list_of_args)`                  | `aModule.a_function(*a_list_of_args)`                       |

1. 最简单的形式，可以通过在参数列表(就像`[a, b, c]`一样)前添加一个星号来调用函数。这跟Python 2里的`apply()`函数是等价的。
2. 在Python 2里，`apply()`函数实际上可以带3个参数：一个函数，一个参数列表，一个字典命名参数(dictionary of named arguments)。在Python 3里，你可以通过在参数列表前添加一个星号(`*`)，在字典命名参数前添加两个星号(`**`)来达到同样的效果。
3. 运算符`+`在这里用作连接列表的功能，它的优先级高于运算符`*`，所以没有必要在`a_list_of_args + z`周围添加额外的括号。
4. `2to3`脚本足够智能来转换复杂的`apply()`调用，包括调用导入模块里的函数。



## 废弃类差异

### <> 比较运算符
Python 2支持`<>`作为`!=`的同义词。Python 3只支持`!=`，不再支持<>了。

| Notes | Python 2          | Python 3          |
| ----- | ----------------- | ----------------- |
| ①     | `if x <> y:`      | `if x != y:`      |
| ②     | `if x <> y <> z:` | `if x != y != z:` |

1. 简单地比较。
2. 相对复杂的三个值之间的比较。

### 字典类方法`has_key()`

在Python 2里，字典对象的`has_key()`方法用来测试字典是否包含特定的键(key)。Python 3不再支持这个方法了。你需要使用`in`运算符。

| Notes | Python 2                                             | Python 3                                 |
| ----- | ---------------------------------------------------- | ---------------------------------------- |
| ①     | `a_dictionary.has_key('PapayaWhip')`                 | `'PapayaWhip' in a_dictionary`           |
| ②     | `a_dictionary.has_key(x) or a_dictionary.has_key(y)` | `x in a_dictionary or y in a_dictionary` |
| ③     | `a_dictionary.has_key(x or y)`                       | `(x or y) in a_dictionary`               |
| ④     | `a_dictionary.has_key(x + y)`                        | `(x + y) in a_dictionary`                |
| ⑤     | `x + a_dictionary.has_key(y)`                        | `x + (y in a_dictionary)`                |

1. 最简单的形式。
2. 运算符`or`的优先级低于运算符`in`，所以这里不需要添加括号。
3. 另一方面，出于同样的原因 — `or`的优先级低于`in`，这里需要添加括号。(注意：这里的代码与前面那行完全不同。Python会先解释`x or y`，得到结果x(如果x在布尔上下文里的值是真)或者y。然后Python检查这个结果是不是a_dictionary的一个键。)
4. 运算符`in`的优先级低于运算符`+`，所以代码里的这种形式从技术上说不需要括号，但是`2to3`还是添加了。
5. 这种形式一定需要括号，因为`in`的优先级低于`+`

### 迭代器方法`next()`

在Python 2里，迭代器有一个`next()`方法，用来返回序列里的下一项。在Python 3里这同样成立，但是现在有了一个新的全局的函数`next()`，它使用一个迭代器作为参数。

| Notes | Python 2                                                     | Python 3                                                     |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ①     | `anIterator.next()`                                          | `next(anIterator)`                                           |
| ②     | `a_function_that_returns_an_iterator().next()`               | `next(a_function_that_returns_an_iterator())`                |
| ③     | `class A:    def next(self):        pass`                    | `class A:    def __next__(self):        pass`                |
| ④     | `class A:    def next(self, x, y):        pass`              | *no change*                                                  |
| ⑤     | `next = 42for an_iterator in a_sequence_of_iterators:    an_iterator.next()` | `next = 42for an_iterator in a_sequence_of_iterators:    an_iterator.__next__()` |

1. 最简单的例子，你不再调用一个迭代器的`next()`方法，现在你将迭代器自身作为参数传递给全局函数`next()`。
2. 假如你有一个返回值是迭代器的函数，调用这个函数然后把结果作为参数传递给`next()`函数。(`2to3`脚本足够智能以正确执行这种转换。)
3. 假如你想定义你自己的类，然后把它用作一个迭代器，在Python 3里，你可以通过定义特殊方法`__next__()`来实现。
4. 如果你定义的类里刚好有一个`next()`，它使用一个或者多个参数，`2to3`执行的时候不会动它。这个类不能被当作迭代器使用，因为它的`next()`方法带有参数。
5. 这一个有些复杂。如果你恰好有一个叫做next的本地变量，在Python 3里它的优先级会高于全局函数`next()`。在这种情况下，你需要调用迭代器的特别方法`__next__()`来获取序列里的下一个元素。(或者，你也可以重构代码以使这个本地变量的名字不叫next，但是2to3不会为你做这件事。)

### `repr`(反引号)

在Python 2里，为了得到一个任意对象的字符串表示，有一种把对象包装在反引号里(比如``x``)的特殊语法。在Python 3里，这种能力仍然存在，但是你不能再使用反引号获得这种字符串表示了。你需要使用全局函数`repr()`。

| Notes | Python 2               | Python 3                       |
| ----- | ---------------------- | ------------------------------ |
| ①     | ``x``                  | `repr(x)`                      |
| ②     | ``'PapayaWhip' + `2``` | `repr('PapayaWhip' + repr(2))` |

1. 记住，x可以是任何东西 — 一个类，函数，模块，基本数据类型，等等。`repr()`函数可以使用任何类型的参数。
2. 在Python 2里，反引号可以嵌套，导致了这种令人费解的(但是有效的)表达式。`2to3`足够智能以将这种嵌套调用转换到`repr()`函数。

### 全局函数`xrange()`

在Python 2里，有两种方法来获得一定范围内的数字：`range()`，它返回一个列表，还有`range()`，它返回一个迭代器。在Python 3里，`range()`返回迭代器，`xrange()`不再存在了。

| Notes | Python 2                  | Python 3                   |
| ----- | ------------------------- | -------------------------- |
| ①     | `xrange(10)`              | `range(10)`                |
| ②     | `a_list = range(10)`      | `a_list = list(range(10))` |
| ③     | `[i for i in xrange(10)]` | `[i for i in range(10)]`   |
| ④     | `for i in range(10):`     | *no change*                |
| ⑤     | `sum(range(10))`          | *no change*                |

1. 在最简单的情况下，`2to3`会简单地把`xrange()`转换为`range()`。
2. 如果你的Python 2代码使用`range()`，`2to3`不知道你是否需要一个列表，或者是否一个迭代器也行。出于谨慎，`2to3`可能会报错，然后使用`list()`把`range()`的返回值强制转换为列表类型。
3. 如果在列表解析里有`xrange()`函数，就没有必要将其返回值转换为一个列表，因为列表解析对迭代器同样有效。
4. 类似的，`for`循环也能作用于迭代器，所以这里也没有改变任何东西。
5. 函数`sum()`能作用于迭代器，所以`2to3`也没有在这里做出修改。这同样适用于`min()`，`max()`，`sum()`，list()，`tuple()`，`set()`，`sorted()`，`any()`，`all()`。

### 全局函数`raw_input()`和`input()`

Python 2有两个全局函数，用来在命令行请求用户输入。第一个叫做`input()`，它等待用户输入一个Python表达式(然后返回结果)。第二个叫做`raw_input()`，用户输入什么它就返回什么。这让初学者非常困惑，并且这被广泛地看作是Python语言的一个“肉赘”(wart)。Python 3通过重命名`raw_input()`为`input()`，从而切掉了这个肉赘，所以现在的`input()`就像每个人最初期待的那样工作。

| Notes | Python 2              | Python 3          |
| ----- | --------------------- | ----------------- |
| ①     | `raw_input()`         | `input()`         |
| ②     | `raw_input('prompt')` | `input('prompt')` |
| ③     | `input()`             | `eval(input())`   |

1. 最简单的形式，`raw_input()`被替换成`input()`。
2. 在Python 2里，`raw_input()`函数可以指定一个提示符作为参数。Python 3里保留了这个功能。
3. 如果你真的想要请求用户输入一个Python表达式，计算结果，可以通过调用`input()`函数然后把返回值传递给`eval()`。

### `StandardError`异常

在Python 2里，`StandardError`是除了`StopIteration`，`GeneratorExit`，`KeyboardInterrupt`，`SystemExit`之外所有其他内置异常的基类。在Python 3里，`StandardError`已经被取消了；使用`Exception`替代。

| Notes | Python 2                     | Python 3                 |
| ----- | ---------------------------- | ------------------------ |
|       | `x = StandardError()`        | `x = Exception()`        |
|       | `x = StandardError(a, b, c)` | `x = Exception(a, b, c)` |

### `basestring`数据类型

Python 2有两种字符串类型：Unicode编码的字符串和非Unicode编码的字符串。但是其实还有另外 一种类型，即`basestring`。它是一个抽象数据类型，是`str`和`unicode`类型的超类(superclass)。它不能被直接调用或者实例化，但是你可以把它作为`isinstance()`的参数来检测一个对象是否是一个Unicode字符串或者非Unicode字符串。在Python 3里，只有一种字符串类型，所以`basestring`就没有必要再存在了。

| Notes | Python 2                    | Python 3             |
| ----- | --------------------------- | -------------------- |
|       | `isinstance(x, basestring)` | `isinstance(x, str)` |



## 修改类差异

### 返回列表的字典类方法

在Python 2里，许多字典类方法的返回值是列表。其中最常用方法的有`keys`，`items`和`values`。在Python 3里，所有以上方法的返回值改为动态视图(dynamic view)。在一些上下文环境里，这种改变并不会产生影响。如果这些方法的返回值被立即传递给另外一个函数，并且那个函数会遍历整个序列，那么以上方法的返回值是列表或者视图并不会产生什么不同。在另外一些情况下，Python 3的这些改变干系重大。如果你期待一个能被独立寻址元素的列表，那么Python 3的这些改变将会使你的代码卡住(choke)，因为视图(view)不支持索引(indexing)。

| Notes | Python 2                               | Python 3                           |
| ----- | -------------------------------------- | ---------------------------------- |
| ①     | `a_dictionary.keys()`                  | `list(a_dictionary.keys())`        |
| ②     | `a_dictionary.items()`                 | `list(a_dictionary.items())`       |
| ③     | `a_dictionary.iterkeys()`              | `iter(a_dictionary.keys())`        |
| ④     | `[i for i in a_dictionary.iterkeys()]` | `[i for i in a_dictionary.keys()]` |
| ⑤     | `min(a_dictionary.keys())`             | *no change*                        |

1. 使用`list()`函数将`keys()`的返回值转换为一个静态列表，出于安全方面的考量，`2to3`可能会报错。这样的代码是有效的，但是对于使用视图来说，它的效率低一些。你应该检查转换后的代码，看看是否一定需要列表，也许视图也能完成同样的工作。
2. 这是另外一种视图(关于`items()`方法的)到列表的转换。`2to3`对`values()`方法返回值的转换也是一样的。
3. Python 3里不再支持`iterkeys()`了。如果必要，使用`iter()`将`keys()`的返回值转换成为一个迭代器。
4. `2to3`能够识别出`iterkeys()`方法在列表解析里被使用，然后将它转换为Python 3里的`keys()`方法(不需要使用额外的`iter()`去包装其返回值)。这样是可行的，因为视图是可迭代的。
5. `2to3`也能识别出`keys()`方法的返回值被立即传给另外一个会遍历整个序列的函数，所以也就没有必要先把`keys()`的返回值转换到一个列表。相反的，`min()`函数会很乐意遍历视图。这个过程对`min()`，`max()`，`sum()`，`list()`，`tuple()`，`set()`，`sorted()`，`any()`和`all()`同样有效。

### 被重命名或者重新组织的模块

从Python 2到Python 3，标准库里的一些模块已经被重命名了。还有一些相互关联的模块也被组合或者重新组织，以使得这种关联更有逻辑性。

#### `http`

在Python 3里，几个相关的http模块被组合成一个单独的包，即`http`。

| Notes | Python 2                                                     | Python 3                |
| ----- | ------------------------------------------------------------ | ----------------------- |
| ①     | `import httplib`                                             | `import http.client`    |
| ②     | `import Cookie`                                              | `import http.cookies`   |
| ③     | `import cookielib`                                           | `import http.cookiejar` |
| ④     | `import BaseHTTPServer import SimpleHTTPServer import CGIHttpServer` | `import http.server`    |

1. `http.client`模块实现了一个底层的库，可以用来请求http资源，解析http响应。
2. `http.cookies`模块提供一个蟒样的(Pythonic)接口来获取通过http头部(http header)Set-Cookie发送的cookies
3. 常用的流行的浏览器会把cookies以文件形式存放在磁盘上，`http.cookiejar`模块可以操作这些文件。
4. `http.server`模块实现了一个基本的http服务器

#### `urllib`

Python 2有一些用来分析，编码和获取URL的模块，但是这些模块就像老鼠窝一样相互重叠。在Python 3里，这些模块被重构、组合成了一个单独的包，即`urllib`。

| Notes | Python 2                                                     | Python 3                                                     |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ①     | `import urllib`                                              | `import urllib.request, urllib.parse, urllib.error`          |
| ②     | `import urllib2`                                             | `import urllib.request, urllib.error`                        |
| ③     | `import urlparse`                                            | `import urllib.parse`                                        |
| ④     | `import robotparser`                                         | `import urllib.robotparser`                                  |
| ⑤     | `from urllib import FancyURLopenerfrom urllib import urlencode` | `from urllib.request import FancyURLopenerfrom urllib.parse import urlencode` |
| ⑥     | `from urllib2 import Requestfrom urllib2 import HTTPError`   | `from urllib.request import Requestfrom urllib.error import HTTPError` |

1. 以前，Python 2里的`urllib`模块有各种各样的函数，包括用来获取数据的`urlopen()`，还有用来将url分割成其组成部分的`splittype()`，`splithost()`和`splituser()`函数。在新的`urllib`包里，这些函数被组织得更有逻辑性。2to3将会修改这些函数的调用以适应新的命名方案。
2. 在Python 3里，以前的`urllib2`模块被并入了`urllib`包。同时，以`urllib2`里各种你最喜爱的东西将会一个不缺地出现在Python 3的`urllib`模块里，比如`build_opener()`方法，`Request`对象，`HTTPBasicAuthHandler`和friends。
3. Python 3里的`urllib.parse`模块包含了原来Python 2里`urlparse`模块所有的解析函数。
4. `urllib.robotparse`模块解析[`robots.txt`文件。
5. 处理http重定向和其他状态码的`FancyURLopener`类在Python 3里的`urllib.request`模块里依然有效。`urlencode()`函数已经被转移到了`urllib.parse`里。
6. `Request`对象在`urllib.request`里依然有效，但是像`HTTPError`这样的常量已经被转移到了`urllib.error`里。

我是否有提到`2to3`也会重写你的函数调用？比如，如果你的Python 2代码里导入了`urllib`模块，调用了`urllib.urlopen()`函数获取数据，`2to3`会同时修改`import`语句和函数调用。

| Notes | Python 2                                                     | Python 3                                                     |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|       | `import urllibprint urllib.urlopen('http://diveintopython3.org/').read()` | `import urllib.request, urllib.parse, urllib.errorprint(urllib.request.urlopen('http://diveintopython3.org/').read())` |

#### `dbm`

所有的dbm克隆(dbm clone)现在在单独的一个包里，即`dbm`。如果你需要其中某个特定的变体，比如gnu dbm，你可以导入`dbm`包中合适的模块。

| Notes | Python 2                      | Python 3          |
| ----- | ----------------------------- | ----------------- |
|       | `import dbm`                  | `import dbm.ndbm` |
|       | `import gdbm`                 | `import dbm.gnu`  |
|       | `import dbhash`               | `import dbm.bsd`  |
|       | `import dumbdbm`              | `import dbm.dumb` |
|       | `import anydbmimport whichdb` | `import dbm`      |

#### `xmlrpc`

xml-rpc是一个通过http协议执行远程rpc调用的轻重级方法。一些xml-rpc客户端和xml-rpc服务端的实现库现在被组合到了独立的包，即`xmlrpc`。

| Notes | Python 2                                          | Python 3               |
| ----- | ------------------------------------------------- | ---------------------- |
|       | `import xmlrpclib`                                | `import xmlrpc.client` |
|       | `import DocXMLRPCServerimport SimpleXMLRPCServer` | `import xmlrpc.server` |

#### 其他模块

| Notes | Python 2                                                     | Python 3              |
| ----- | ------------------------------------------------------------ | --------------------- |
| ①     | `try:    import cStringIO as StringIO except ImportError:    import StringIO` | `import io`           |
| ②     | `try:    import cPickle as pickle except ImportError:    import pickle` | `import pickle`       |
| ③     | `import __builtin__`                                         | `import builtins`     |
| ④     | `import copy_reg`                                            | `import copyreg`      |
| ⑤     | `import Queue`                                               | `import queue`        |
| ⑥     | `import SocketServer`                                        | `import socketserver` |
| ⑦     | `import ConfigParser`                                        | `import configparser` |
| ⑧     | `import repr`                                                | `import reprlib`      |
| ⑨     | `import commands`                                            | `import subprocess`   |

1. 在Python 2里，你通常会这样做，首先尝试把`cStringIO`导入作为`StringIO`的替代，如果失败了，再导入`StringIO`。不要在Python 3里这样做；`io`模块会帮你处理好这件事情。它会找出可用的最快实现方法，然后自动使用它。
2. 在Python 2里，导入最快的`pickle`实现也是一个与上边相似的能用方法。在Python 3里，`pickle`模块会自动为你处理，所以不要再这样做。
3. `builtins`模块包含了在整个Python语言里都会使用的全局函数，类和常量。重新定义`builtins`模块里的某个函数意味着在每处都重定义了这个全局函数。这听起来很强大，但是同时也是很可怕的。
4. `copyreg`模块为用C语言定义的用户自定义类型添加了`pickle`模块的支持。
5. `queue`模块实现一个生产者消费者队列(multi-producer, multi-consumer queue)。
6. `socketserver`模块为实现各种socket server提供了通用基础类。
7. `configparser`模块用来解析ini-style配置文件。
8. `reprlib`模块重新实现了内置函数`repr()`，并添加了对字符串表示被截断前长度的控制。
9. `subprocess`模块允许你创建子进程，连接到他们的管道，然后获取他们的返回值。



### 包内的相对导入

包是由一组相关联的模块共同组成的单个实体。在Python 2的时候，为了实现同一个包内模块的相互引用，你会使用`import foo`或者`from foo import Bar`。Python 2解释器会先在当前目录里搜索`foo.py`，然后再去Python搜索路径(`sys.path`)里搜索。在Python 3里这个过程有一点不同。Python 3不会首先在当前路径搜索，它会直接在Python的搜索路径里寻找。如果你想要包里的一个模块导入包里的另外一个模块，你需要显式地提供两个模块的相对路径。

假设你有如下包，多个文件在同一个目录下：

```
chardet/
|
+--__init__.py
|
+--constants.py
|
+--mbcharsetprober.py
|
+--universaldetector.py
```

现在假设`universaldetector.py`需要整个导入`constants.py`，另外还需要导入`mbcharsetprober.py`的一个类。你会怎样做?

| Notes | Python 2                                             | Python 3                                              |
| ----- | ---------------------------------------------------- | ----------------------------------------------------- |
| ①     | `import constants`                                   | `from . import constants`                             |
| ②     | `from mbcharsetprober import MultiByteCharSetProber` | `from .mbcharsetprober import MultiByteCharsetProber` |

1. 当你需要从包的其他地方导入整个模块，使用新的`from . import`语法。这里的句号(.)即表示当前文件(`universaldetector.py`)和你想要导入文件(`constants.py`)之间的相对路径。在这个样例中，这两个文件在同一个目录里，所以使用了单个句号。你也可以从父目录(`from .. import anothermodule`)或者子目录里导入。
2. 为了将一个特定的类或者函数从其他模块里直接导入到你的模块的名字空间里，在需要导入的模块名前加上相对路径，并且去掉最后一个斜线(slash)。在这个例子中，`mbcharsetprober.py`与`universaldetector.py`在同一个目录里，所以相对路径名就是一个句号。你也可以从父目录(from .. import anothermodule)或者子目录里导入。

### 全局函数`filter()`

在Python 2里，`filter()`方法返回一个列表，这个列表是通过一个返回值为`True`或者`False`的函数来检测序列里的每一项得到的。在Python 3里，`filter()`函数返回一个迭代器，不再是列表。

| Notes | Python 2                                      | Python 3                               |
| ----- | --------------------------------------------- | -------------------------------------- |
| ①     | `filter(a_function, a_sequence)`              | `list(filter(a_function, a_sequence))` |
| ②     | `list(filter(a_function, a_sequence))`        | *no change*                            |
| ③     | `filter(None, a_sequence)`                    | `[i for i in a_sequence if i]`         |
| ④     | `for i in filter(None, a_sequence):`          | *no change*                            |
| ⑤     | `[i for i in filter(a_function, a_sequence)]` | *no change*                            |

1. 最简单的情况下，`2to3`会用一个`list()`函数来包装`filter()`，`list()`函数会遍历它的参数然后返回一个列表。
2. 然而，如果`filter()`调用已经被`list()`包裹，`2to3`不会再做处理，因为这种情况下`filter()`的返回值是否是一个迭代器是无关紧要的。
3. 为了处理`filter(None, ...)`这种特殊的语法，`2to3`会将这种调用从语法上等价地转换为列表解析。
4. 由于`for`循环会遍历整个序列，所以没有必要再做修改。
5. 与上面相同，不需要做修改，因为列表解析会遍历整个序列，即使`filter()`返回一个迭代器，它仍能像以前的`filter()`返回列表那样正常工作。

### 全局函数`map()`

跟`filter()`作的改变一样，`map()`函数现在返回一个迭代器。(在Python 2里，它返回一个列表。)

| Notes | Python 2                                   | Python 3                              |
| ----- | ------------------------------------------ | ------------------------------------- |
| ①     | `map(a_function, 'PapayaWhip')`            | `list(map(a_function, 'PapayaWhip'))` |
| ②     | `map(None, 'PapayaWhip')`                  | `list('PapayaWhip')`                  |
| ③     | `map(lambda x: x+1, range(42))`            | `[x+1 for x in range(42)]`            |
| ④     | `for i in map(a_function, a_sequence):`    | *no change*                           |
| ⑤     | `[i for i in map(a_function, a_sequence)]` | *no change*                           |

1. 类似对`filter()`的处理，在最简单的情况下，`2to3`会用一个`list()`函数来包装`map()`调用。
2. 对于特殊的`map(None, ...)`语法，跟`filter(None, ...)`类似，`2to3`会将其转换成一个使用`list()`的等价调用
3. 如果`map()`的第一个参数是一个lambda函数，`2to3`会将其等价地转换成列表解析。
4. 对于会遍历整个序列的`for`循环，不需要做改变。
5. 再一次地，这里不需要做修改，因为列表解析会遍历整个序列，即使`map()`的返回值是迭代器而不是列表它也能正常工作。

### 全局函数`reduce()`

在Python 3里，`reduce()`函数已经被从全局名字空间里移除了，它现在被放置在`fucntools`模块里。

| Notes | Python 2          | Python 3                                      |
| ----- | ----------------- | --------------------------------------------- |
|       | `reduce(a, b, c)` | `from functools import reducereduce(a, b, c)` |

### `exec`语句

就像`print`语句在Python 3里变成了一个函数一样，`exec`语句也是这样的。`exec()`函数使用一个包含任意Python代码的字符串作为参数，然后就像执行语句或者表达式一样执行它。`exec()`跟`eval()`是相似的，但是`exec()`更加强大并更具有技巧性。`eval()`函数只能执行单独一条表达式，但是`exec()`能够执行多条语句，导入(import)，函数声明 — 实际上整个Python程序的字符串表示也可以。

| Notes | Python 2                                                   | Python 3                                                  |
| ----- | ---------------------------------------------------------- | --------------------------------------------------------- |
| ①     | `exec codeString`                                          | `exec(codeString)`                                        |
| ②     | `exec codeString in a_global_namespace`                    | `exec(codeString, a_global_namespace)`                    |
| ③     | `exec codeString in a_global_namespace, a_local_namespace` | `exec(codeString, a_global_namespace, a_local_namespace)` |

1. 在最简单的形式下，因为`exec()`现在是一个函数，而不是语句，`2to3`会把这个字符串形式的代码用括号围起来。
2. Python 2里的`exec`语句可以指定名字空间，代码将在这个由全局对象组成的私有空间里执行。Python 3也有这样的功能；你只需要把这个名字空间作为第二个参数传递给`exec()`函数。
3. 更加神奇的是，Python 2里的`exec`语句还可以指定一个本地名字空间(比如一个函数里声明的变量)。在Python 3里，`exec()`函数也有这样的功能。

### `try...except`语句

从Python 2到Python 3，捕获异常的语法有些许变化。

| Notes | Python 2                                                     | Python 3                                                     |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ①     | `try:    import mymodule except ImportError, e    pass`      | `try:    import mymoduleexcept ImportError as e:    pass`    |
| ②     | `try:    import mymodule except (RuntimeError, ImportError), e    pass` | `try:    import mymoduleexcept (RuntimeError, ImportError) as e:    pass` |
| ③     | `try:    import mymodule except ImportError:    pass`        | *no change*                                                  |
| ④     | `try:    import mymodule except:    pass`                    | *no change*                                                  |

1. 相对于Python 2里在异常类型后添加逗号，Python 3使用了一个新的关键字，`as`。
2. 关键字`as`也可以用在一次捕获多种类型异常的情况下。
3. 如果你捕获到一个异常，但是并不在意访问异常对象本身，Python 2和Python 3的语法是一样的。
4. 类似地，如果你使用一个保险方法(fallback)来捕获*所有*异常，Python 2和Python 3的语法是一样的。

### `raise`语句

Python 3里，抛出自定义异常的语法有细微的变化。

| Notes | Python 2                                          | Python 3                                                     |
| ----- | ------------------------------------------------- | ------------------------------------------------------------ |
| ①     | `raise MyException`                               | *unchanged*                                                  |
| ②     | `raise MyException, 'error message'`              | `raise MyException('error message')`                         |
| ③     | `raise MyException, 'error message', a_traceback` | `raise MyException('error message').with_traceback(a_traceback)` |
| ④     | `raise 'error message'`                           | *unsupported*                                                |

1. 抛出不带用户自定义错误信息的异常，这种最简单的形式下，语法没有改变。
2. 当你想要抛出一个带用户自定义错误信息的异常时，改变就显而易见了。Python 2用一个逗号来分隔异常类和错误信息；Python 3把错误信息作为参数传递给异常类。
3. Python 2支持一种更加复杂的语法来抛出一个带用户自定义回溯(stack trace，堆栈追踪)的异常。在Python 3里你也可以这样做，但是语法完全不同。
4. 在Python 2里，你可以抛出一个不带异常类的异常，仅仅只有一个异常信息。在Python 3里，这种形式不再被支持。`2to3`将会警告你它不能自动修复这种语法。

### 函数属性`func_*`

在Python 2里，函数的里的代码可以访问到函数本身的特殊属性。在Python 3里，为了一致性，这些特殊属性被重新命名了。

| Notes | Python 2                   | Python 3                  |
| ----- | -------------------------- | ------------------------- |
| ①     | `a_function.func_name`     | `a_function.__name__`     |
| ②     | `a_function.func_doc`      | `a_function.__doc__`      |
| ③     | `a_function.func_defaults` | `a_function.__defaults__` |
| ④     | `a_function.func_dict`     | `a_function.__dict__`     |
| ⑤     | `a_function.func_closure`  | `a_function.__closure__`  |
| ⑥     | `a_function.func_globals`  | `a_function.__globals__`  |
| ⑦     | `a_function.func_code`     | `a_function.__code__`     |

1. `__name__`属性(原`func_name`)包含了函数的名字。
2. `__doc__`属性(原`funcdoc`)包含了你在函数源代码里定义的文档字符串(*docstring*)
3. `__defaults__`属性(原`func_defaults`)是一个保存参数默认值的元组。
4. `__dict__`属性(原`func_dict`)是一个支持任意函数属性的名字空间。
5. `__closure__`属性(原`func_closure`)是一个由cell对象组成的元组，它包含了函数对自由变量(free variable)的绑定。
6. `__globals__`属性(原`func_globals`)是一个对模块全局名字空间的引用，函数本身在这个名字空间里被定义。
7. `__code__`属性(原`func_code`)是一个代码对象，表示编译后的函数体。

### 使用元组而非多个参数的`lambda`函数

在Python 2里，你可以定义匿名`lambda`函数，通过指定作为参数的元组的元素个数，使这个函数实际上能够接收多个参数。事实上，Python 2的解释器把这个元组“解开”(unpack)成命名参数(named arguments)，然后你可以在`lambda`函数里引用它们(通过名字)。在Python 3里，你仍然可以传递一个元组作为`lambda`函数的参数，但是Python解释器不会把它解析成命名参数。你需要通过位置索引(positional index)来引用每个参数。

| Notes | Python 2                        | Python 3                                             |
| ----- | ------------------------------- | ---------------------------------------------------- |
| ①     | `lambda (x,): x + f(x)`         | `lambda x1: x1[0] + f(x1[0])`                        |
| ②     | `lambda (x, y): x + f(y)`       | `lambda x_y: x_y[0] + f(x_y[1])`                     |
| ③     | `lambda (x, (y, z)): x + y + z` | `lambda x_y_z: x_y_z[0] + x_y_z[1][0] + x_y_z[1][1]` |
| ④     | `lambda x, y, z: x + y + z`     | *unchanged*                                          |

1. 如果你已经定义了一个`lambda`函数，它使用包含一个元素的元组作为参数，在Python 3里，它会被转换成一个包含到x1[0]的引用的`lambda`函数。x1是`2to3`脚本基于原来元组里的命名参数自动生成的。
2. 使用含有两个元素的元组(x, y)作为参数的`lambda`函数被转换为x_y，它有两个位置参数，即x_y[0]和x_y[1]。
3. `2to3`脚本甚至可以处理使用嵌套命名参数的元组作为参数的`lambda`函数。产生的结果代码有点难以阅读，但是它在Python 3下跟原来的代码在Python 2下的效果是一样的。
4. 你可以定义使用多个参数的`lambda`函数。如果没有括号包围在参数周围，Python 2会把它当作一个包含多个参数的`lambda`函数；在这个`lambda`函数体里，你通过名字引用这些参数，就像在其他类型的函数里所做的一样。这种语法在Python 3里仍然有效。

### 八进制类型

在Python 2和Python 3之间，定义八进制(octal)数的语法有轻微的改变。

| Notes | Python 2   | Python 3    |
| ----- | ---------- | ----------- |
|       | `x = 0755` | `x = 0o755` |

## `sys.maxint`

由于长整型和整型被整合在一起了，`sys.maxint`常量不再精确。但是因为这个值对于检测特定平台的能力还是有用处的，所以它被Python 3保留，并且重命名为`sys.maxsize`。

| Notes | Python 2                 | Python 3                  |
| ----- | ------------------------ | ------------------------- |
| ①     | `from sys import maxint` | `from sys import maxsize` |
| ②     | `a_function(sys.maxint)` | `a_function(sys.maxsize)` |

1. `maxint`变成了`maxsize`。
2. 所有的`sys.maxint`都变成了`sys.maxsize`。

### 全局函数`zip()`

在Python 2里，全局函数`zip()`可以使用任意多个序列作为参数，它返回一个由元组构成的列表。第一个元组包含了每个序列的第一个元素；第二个元组包含了每个序列的第二个元素；依次递推下去。在Python 3里，`zip()`返回一个迭代器，而非列表。

| Notes | Python 2               | Python 3             |
| ----- | ---------------------- | -------------------- |
| ①     | `zip(a, b, c)`         | `list(zip(a, b, c))` |
| ②     | `d.join(zip(a, b, c))` | *no change*          |

1. 最简单的形式，你可以通过调用`list()`函数包装`zip()`的返回值来恢复`zip()`函数以前的功能，`list()`函数会遍历这个`zip()`函数返回的迭代器，然后返回结果的列表表示。
2. 在已经会遍历序列所有元素的上下文环境里(比如这里对`join()`方法的调用)，`zip()`返回的迭代器能够正常工作。`2to3`脚本会检测到这些情况，不会对你的代码作出改变。

### 对元组的列表解析

在Python 2里，如果你需要编写一个遍历元组的列表解析，你不需要在元组值的周围加上括号。在Python 3里，这些括号是必需的。

| Notes | Python 2            | Python 3              |
| ----- | ------------------- | --------------------- |
|       | `[i for i in 1, 2]` | `[i for i in (1, 2)]` |

### 元类(metaclass)

在Python 2里，你可以通过在类的声明中定义`metaclass`参数，或者定义一个特殊的类级别的(class-level)`__metaclass__`属性，来创建元类。在Python 3里，`__metaclass__`属性已经被取消了。

| Notes | Python 2                                                  | Python 3                                                  |
| ----- | --------------------------------------------------------- | --------------------------------------------------------- |
| ①     | `class C(metaclass=PapayaMeta):    pass`                  | *unchanged*                                               |
| ②     | `class Whip:    __metaclass__ = PapayaMeta`               | `class Whip(metaclass=PapayaMeta):    pass`               |
| ③     | `class C(Whipper, Beater):    __metaclass__ = PapayaMeta` | `class C(Whipper, Beater, metaclass=PapayaMeta):    pass` |

1. 在声明类的时候声明`metaclass`参数，这在Python 2和Python 3里都有效，它们是一样的。
2. 在类的定义里声明`__metaclass__`属性在Python 2里有效，但是在Python 3里不再有效。
3. `2to3`能够构建一个有效的类声明，即使这个类继承自多个父类。



------

#### 参考来源

[Python 2 vs Python 3: Practical Considerations](https://www.digitalocean.com/community/tutorials/python-2-vs-python-3-practical-considerations-2)

[使用`2to3`将代码移植到Python 3](https://woodpecker.org.cn/diveintopython3/porting-code-to-python-3-with-2to3.html)

#### 更多

[2to3-Automated Python 2 to 3 code translation](https://docs.python.org/2/library/2to3.html)

[案例研究:将`chardet`移植到Python 3](https://woodpecker.org.cn/diveintopython3/case-study-porting-chardet-to-python-3.html)