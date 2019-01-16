# github和git简明指南

### 注册账户以及创建仓库

 github官网地址：[https://github.com](https://github.com/)

### Git 安装

- [下载 git OSX 版](http://code.google.com/p/git-osx-installer/downloads/list?can=3)
- [下载 git Windows 版](http://msysgit.github.io/)
- [下载 git Linux 版](http://book.git-scm.com/2_installing_git.html)

### 配置Git

首先在本地创建`ssh key；`

```
$ ssh-keygen -t rsa -C "xxx@xxx.com"
```

后面的`xxx@xxx.com`改为你在github上注册的邮箱，然后一路回车，成功后复制`~/.ssh/id_rsa.pub`里面的key，粘贴到github上。

然后输入如下验证是否成功：

```
$ ssh -T git@github.com
```

接下来我们要做的就是把本地仓库传到github上去，在此之前还需要设置username和email，因为github每次commit都会记录他们。

```
$ git config --global user.name "your name"
$ git config --global user.email "xxx@xxx.com"
```

### 检出仓库

有如下两种方式：

- 克隆

  ```
  git clone git@github.com:yourName/yourRepo.git
  ```

  `git@github.com:yourName/yourRepo.git` 为你的远程仓库地址。

- 本地初始化

  创建新文件夹，进入，然后执行` git init` 创建新的 git 仓库并添加远程地址：

  ```
  $ git remote add origin git@github.com:yourName/yourRepo.git
  ```

然后你就可以使用`git push origin master`将你的文件推送到远程仓库了，这里master* 是"默认的"分支。

### 工作流

你的本地仓库由 git 维护的三棵"树"组成。第一个是你的 `工作目录`，它持有实际文件；第二个是 `暂存区（Index）`，它像个缓存区域，临时保存你的改动；最后是 `HEAD`，它指向你最后一次提交的结果。

你可以提出更改（把它们添加到暂存区），使用如下命令：
`git add <filename>`
`git add *`
这是 git 基本工作流程的第一步；使用如下命令以实际提交改动：
`git commit -m "代码提交信息"`
现在，你的改动已经提交到了 **HEAD**，但是还没到你的远端仓库。

另外，可以通过`git status` 查看当前仓库当前状态。

### 分支

分支是用来将特性开发绝缘开来的。在其他分支上进行开发，完成后再将它们合并到主分支上。

创建一个叫做"feature_x"的分支，并切换过去：
`git checkout -b feature_x`
切换回主分支：
`git checkout master`
再把新建的分支删掉：
`git branch -d feature_x`
除非你将分支推送到远端仓库，不然该分支就是 *不为他人所见的*：
`git push origin <branch>`

### 更新与合并

要更新你的本地仓库至最新改动，执行：
`git pull`
以在你的工作目录中 *获取（fetch）* 并 *合并（merge）* 远端的改动。
要合并其他分支到你的当前分支（例如 master），执行：
`git merge <branch>`
在这两种情况下，git 都会尝试去自动合并改动。遗憾的是，这可能并非每次都成功，并可能出现*冲突（conflicts）*。 这时候就需要你修改这些文件来手动合并这些*冲突（conflicts）*。改完之后，你需要执行如下命令以将它们标记为合并成功：
`git add <filename>`
在合并改动之前，你可以使用如下命令预览差异：
`git diff <source_branch> <target_branch>`



### 更新与合并

要更新你的本地仓库至最新改动，执行：
`git pull`
以在你的工作目录中 *获取（fetch）* 并 *合并（merge）* 远端的改动。
要合并其他分支到你的当前分支（例如 master），执行：
`git merge <branch>`
在这两种情况下，git 都会尝试去自动合并改动。遗憾的是，这可能并非每次都成功，并可能出现*冲突（conflicts）*。 这时候就需要你修改这些文件来手动合并这些*冲突（conflicts）*。改完之后，你需要执行如下命令以将它们标记为合并成功：
`git add <filename>`
在合并改动之前，你可以使用如下命令预览差异：
`git diff <source_branch> <target_branch>`

### 标签

你可以执行如下命令创建一个叫做 *1.0.0* 的标签：
`git tag 1.0.0 303a9c113e`
*303a9c113e* 是你想要标记的提交 ID 的前 10 位字符。可以使用下列命令获取提交 ID：
`git log`
你也可以使用少一点的提交 ID 前几位，只要它的指向具有唯一性。

### 替换本地改动

假如你操作失误（当然，这最好永远不要发生），你可以使用如下命令替换掉本地改动：
`git checkout -- <filename>`
此命令会使用 HEAD 中的最新内容替换掉你的工作目录中的文件。已添加到暂存区的改动以及新文件都不会受到影响。

假如你想丢弃你在本地的所有改动与提交，可以到服务器上获取最新的版本历史，并将你本地主分支指向它：
`git fetch origin`
`git reset --hard origin/master`

### git更多

[Pro Git 第二版](https://git-scm.com/book/zh/v2)  --Git 学习的圣经

[githug](https://github.com/Gazler/githug) --通过游戏点亮git技能

