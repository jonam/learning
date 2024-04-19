## Uncommon git commands

Show log of commits with date.

```
git log --format="%h %ad | %s" --date=short
```

## How to remove local commits upto a hash

Let's say your history looks like this, and you need to roll back to hash bb7bd34. 

Then you can do, following which will bring you to this hash bb7bd34, meaning the commit for this hash will not be dropped.

```
git reset --hard bb7bd34
```


```
git log --format="%h %ad | %s" --date=short
```

```
3844540 2024-02-21 | some ai utils
6e28f6b 2024-02-03 | committing changes
9207590 2023-12-07 | leet16
bb7bd34 2023-12-07 | leet33
58b374b 2023-12-07 | leet153
83bbe3a 2023-12-06 | leet 152
fa61189 2023-12-06 | leet 152
d41be21 2023-12-06 | leet 53
3980f14 2023-12-06 | leet 238
41e3ba9 2023-12-06 | leet 217
aeaac04 2023-12-06 | leet 121
dbd2c1f 2023-12-06 | renamed file
a04ef2f 2023-12-06 | two_sum
8102196 2023-11-26 | adding bomb squad
f81feaa 2023-10-25 | other questions
351f5d0 2023-10-25 | adding some interview questions for system design
00039c1 2023-09-26 | commit a variety of programs
68d1e7b 2023-09-26 | adding subnet, network
04e553b 2023-09-26 | initial info
1b8d2b0 2023-09-26 | Initial commit
```

But let's say you made a mistake deleting your commits. You can still recover your changes by using:

```
git checkout -b newbranch 3844540
```

Then you can backup your code just in case to some directory outside the git tree. After that you can:

```
git checkout main
git merge newbranch
```

## Permission issues

git commit -m "azure arch"
[main 3224ba3] azure arch
 1 file changed, 20 insertions(+)
 create mode 100644 azure/azure_arch.md
bla@bli azure % git push
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
bla@bli azure % eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github_25519

Agent pid 45797
Identity added: /Users/bla/.ssh/github_25519 (bla.bla@gmail.com)
bla@bli azure % git push              
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Delta compression using up to 8 threads
Compressing objects: 100% (3/3), done.
Writing objects: 100% (4/4), 516 bytes | 516.00 KiB/s, done.
Total 4 (delta 1), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (1/1), completed with 1 local object.
To github.com:bla/learning.git
   66e4ee6..3224ba3  main -> main
