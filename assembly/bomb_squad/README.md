# 1.0 Getting Started

Congratulations! It looks like you have decided to participate in the Bomb Squad!
This section will detail what tools you need to play the Bomb Squad game.

## 1.1 Windows x86 Users

You will need the following tools:

### 1.1.1 Windows: MinGW C/C++ for Windows/Windows 10

Download MinGW C/C++ from here:

https://osdn.net/projects/mingw/

This will most likely be installed in

```
c:\msys64\mingw64\bin
```

but please verify how your PATH variable is updated.

Please note that the directory will be in your PATH variable. For example:

```
>echo %PATH%

C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Windows\System32\OpenSSH\;C:\Program Files\Amazon\cfn-bootstrap\;C:\Program Files\Microsoft VS Code\bin;C:\msys64\mingw64\bin;C:\ProgramData\chocolatey\bin;C:\Users\Administrator\AppData\Local\Microsoft\WindowsApps;
```

### 1.1.2 Windows: Download gdb

If MinGW does not automatically download the GNU debugger (gdb.exe), you can specifically download it:

http://www.equation.com/servlet/equation.cmd?fa=gdb#google_vignette

And for your windows platform download the appropriate 32-bit or 64-bit binary. 

Copy this gdb.exe to the mingw64\bin directory as below:

```
copy c:\Users\Administrator\Downloads\gdb.exe c:\msys64\mingw64\bin
```

### 1.1.3 Windows: Download bomb_squad.exe

You can download the bomb_squad.exe client from here:

## 1.2 Mac Users (x86 and ARM64)

### 1.2.1 Mac: Download Xcode For Mac

If not already downloaded, you may want to download xcode or upgrade it.

https://developer.apple.com/download/

This should include everything you need including clang/gcc and lldb/gdb. We recommend you use lldb as the debugger as that is the new debugger supported on Macs. You may want to choose the Command Line Tools option as all debugging you do will be at the command line. Visual Studio and other IDE tools may not work well with assembly level debugging.

### 1.2.2 Mac: Download bomb_squad client

You can download the bomb_squad client from here. Make sure you download the appropriate client based on whether you have an x86 Mac or an ARM64 Mac (newer macs).

#### 1.2.2.1 Mac x86 Users: Download bomb_squad client

You can download the x86 bomb_squad client from here:

#### 1.2.2.2 Mac ARM64 Users: Download bomb_squad client

You can download the ARM64 bomb_squad client from here:

## 1.3 Linux x86 Users (Ubuntu/Other Linux)

Please note this client is mainly tested on Ubuntu x86 and it should work on other Linux flavors but use them at your own risk.

### 1.3.1 Download C/C++ compiler and debugger

In most cases you probably already have gcc/clang as your C compilers installed, and you also have gdb/lldb as your C debuggers which is all you need. In case you do not have them, here is how you install them:

```
sudo apt update
sudo apt install build-essential
sudo apt-get install gdb
```

You only need the pre-built binaries.

### 1.3.2 Ubuntu x86 Users: Download bomb_squad client

You can download the x86 bomb_squad client from here:

# 2.1 Source/Dest vs Dest/Source Confusion

There are two kinds of syntax:

<b>AT&T Syntax</b>: You can tell that it is AT&T syntax by the presence of little symbols, like the $ preceding literals, and the % preceding registers. For example, this instruction:

```
sub    $16, %rax
```

uses the AT&T syntax. It subtracts 16 from the value in the rax register, and stores the result back in rax.

In AT&T syntax, the destination operand is on the right:

```
insn   source, destination     # AT&T syntax
```

<b>Intel Syntax</b>: This is ubiquitous on Windows platforms, and usually also available as an option for Gnu/Linux tools. Intel syntax is unadorned—e.g.:

```
insn  destination, source     ; Intel syntax
```

To be absolutely certain of which version you've got, you'd need to check the settings for your disassembler/debugger and see what syntax it is configured to use.

You can change that in gdb using:

set disassembly-flavor intel

To make the change permanent you must add it to your gdb config file:

```
echo "set disassembly-flavor intel" >> ~/.gdbinit
```

# Some Frequently Used Registers

* rax is the 64-bit, "long" size register.  It was added in 2003 during the transition to 64-bit processors.
* eax is the 32-bit, "int" size register.  It was added in 1985 during the transition to 32-bit processors with the 80386 CPU. They also work in 32 bit mode.

More details here: 

- https://www.cs.uaf.edu/2017/fall/cs301/lecture/09_11_registers.html
- https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf

# Instruction Suffixes

Instructions are suffixed with the letters "b", "s", "w", "l", "q" or "t" to determine what size operand is being manipulated.

* b = byte (8 bit)
* s = short (16 bit integer) or single (32-bit floating point)
* w = word (16 bit)
* l = long (32 bit integer or 64-bit floating point)
* q = quad (64 bit)
* t = ten bytes (80-bit floating point)

# Life Saving Commands

## Main Tip

First and foremost, install and run a program called strings on your executable. This is usually found on Ubuntu or Mac.

On windows you can try using https://learn.microsoft.com/en-us/sysinternals/downloads/strings but it's not identical.

## Assembly and gdb tips

### Before You Begin Debugging

Please note that the bomb_squad client is partially compiled with debugging information. If you did have all the debugging information, it would be no fun to play the game, as you can just display the source and read where the passcodes are! Specifically the chamber open_door function module is compiled without debugging information. This simply means that you will only be able to see the assembly code and figure out (reverse engineer) the passcodes by playing with the values in the registers. In the subsequent sections we will illustrate how you can use the "disass" command to disassemble the code, so that you can see this assembly in action.

### Important Reference

Almost everything you need for the assembly debugging can be found in these sections of the GDB manual.

Disassembly:

```
https://sourceware.org/gdb/current/onlinedocs/gdb/Machine-Code.html#Machine-Code
```

Examining Memory:

```
https://sourceware.org/gdb/current/onlinedocs/gdb/Memory.html#Memory
```

### Basics

In order to start debugging, you would have downloaded the bomb_squad executable. For instance on windows:

```
gdb .\bomb_squad.exe
```

Once you are in the debugger you can use the list command to list N lines:

```
l 70
```

Above, N is 70 lines. Now let's you want to break at the first chamber door, which would be something you may want to do frequently to save time:

```
(gdb) b 65
Breakpoint 1 at 0x140001747: file bomb_squad.c, line 65.
```

Once you have set the breakpoint you can then connect to our bomb_server. Assuming the IP address or host of the server is 54.219.164.216:

```
(gdb) r 54.219.164.216
Starting program: C:\Users\Administrator\Documents\bomb\bomb_squad.exe 54.219.164.216
```

### Step In vs Step Over

You can refer to this section for some details:

```
https://sourceware.org/gdb/current/onlinedocs/gdb/Continuing-and-Stepping.html#Continuing-and-Stepping
```

In order to step into a function (without debugging information) you can use:

```
si
```

In order to step on a function you can use:

```
s
```

In case you wanted to step into open_door1, you will need to use "si" as most of the chamber code does not come with source code symbols. So you will have access only to assembly code.

### How to show the assembly code

You can use the command:

```
disass
```

This should show you the assembly code.

If you want to disassemble a specific function, you can use:

```
disass open_door1
```

This should show you all the assembly code for the function.

```
Dump of assembler code for function open_door1:
   0x0000000000403070 <+0>:	push   rbp
   0x0000000000403071 <+1>:	mov    rbp,rsp
   0x0000000000403074 <+4>:	sub    rsp,0x30
   0x0000000000403078 <+8>:	mov    QWORD PTR [rbp-0x8],rdi
   0x000000000040307c <+12>:	mov    DWORD PTR [rbp-0xc],esi
   0x000000000040307f <+15>:	call   0x401d20 <msg1>
   0x0000000000403084 <+20>:	call   0x402160 <get_passcode>
   0x0000000000403089 <+25>:	mov    QWORD PTR [rbp-0x18],rax
   0x000000000040308d <+29>:	mov    rdi,QWORD PTR [rbp-0x18]
   0x0000000000403091 <+33>:	call   0x401070 <strlen@plt>
   0x0000000000403096 <+38>:	cmp    rax,0x0
   0x000000000040309a <+42>:	jne    0x4030bb <open_door1+75>
   0x00000000004030a0 <+48>:	movabs rdi,0x40596f
   0x00000000004030aa <+58>:	mov    al,0x0
   0x00000000004030ac <+60>:	call   0x4010b0 <printf@plt>
   0x00000000004030b1 <+65>:	xor    edi,edi
   0x00000000004030b3 <+67>:	mov    DWORD PTR [rbp-0x24],eax
   0x00000000004030b6 <+70>:	call   0x4011e0 <exit@plt>
   0x00000000004030bb <+75>:	cmp    DWORD PTR ds:0x408580,0x0
   0x00000000004030c3 <+83>:	jne    0x403132 <open_door1+194>
   0x00000000004030c9 <+89>:	mov    rdi,QWORD PTR [rbp-0x18]
   0x00000000004030cd <+93>:	mov    esi,0x4059a6
   0x00000000004030d2 <+98>:	call   0x401110 <strcmp@plt>
   0x00000000004030d7 <+103>:	mov    DWORD PTR [rbp-0x1c],eax
   0x00000000004030da <+106>:	mov    rdi,QWORD PTR [rbp-0x8]
   0x00000000004030de <+110>:	mov    esi,DWORD PTR [rbp-0xc]
   0x00000000004030e1 <+113>:	mov    rcx,QWORD PTR [rbp-0x18]
   0x00000000004030e5 <+117>:	mov    r8d,DWORD PTR [rbp-0x1c]
   0x00000000004030e9 <+121>:	mov    edx,0x1
   0x00000000004030ee <+126>:	call   0x402a80 <chamber>
   0x00000000004030f3 <+131>:	mov    DWORD PTR [rbp-0x20],eax
   0x00000000004030f6 <+134>:	cmp    DWORD PTR [rbp-0x1c],0x0
   0x00000000004030fa <+138>:	je     0x403105 <open_door1+149>
   0x0000000000403100 <+144>:	call   0x403010 <door1error>
   0x0000000000403105 <+149>:	cmp    DWORD PTR [rbp-0x20],0xffffffff
   0x0000000000403109 <+153>:	jne    0x40312d <open_door1+189>
   0x000000000040310f <+159>:	movabs rdi,0x4059b3
   0x0000000000403119 <+169>:	mov    al,0x0
   0x000000000040311b <+171>:	call   0x4010b0 <printf@plt>
   0x0000000000403120 <+176>:	mov    edi,0x1
   0x0000000000403125 <+181>:	mov    DWORD PTR [rbp-0x28],eax
   0x0000000000403128 <+184>:	call   0x4011e0 <exit@plt>
   0x000000000040312d <+189>:	jmp    0x403132 <open_door1+194>
   0x0000000000403132 <+194>:	call   0x401da0 <msg2>
   0x0000000000403137 <+199>:	add    rsp,0x30
   0x000000000040313b <+203>:	pop    rbp
   0x000000000040313c <+204>:	ret    
End of assembler dump.
```

### Display Your Current Stack Frame

Use the command

```
frame
```

It will display something like:

```
(gdb) frame
#0  0x00000000004014d1 in main (argc=2, argv=0x7fffffffdca8) at bomb_squad.c:48
48	    open_door1(login, status);
```

### Placing a breakpoint in assembly

Let's say:

```
   0x00007ff69565366c <+37>:    call   0x7ff6956535f0 <scifi>
   0x00007ff695653671 <+42>:    mov    %rax,%rdx
   0x00007ff695653674 <+45>:    mov    -0x8(%rbp),%rax
   0x00007ff695653678 <+49>:    mov    %rax,%rcx
   0x00007ff69565367b <+52>:    call   0x7ff69565a418 <strcmp>
```

you wanted to place a breakpoint at the scifi function. Then you would use:

```
b *0x00007ff69565366c
```

You must make sure you are in the function, where you want to disass the code.


Let's say you look at this instruction (in Intel format):

```
   0x0000000000402de8 <+24>:	mov    rdi,QWORD PTR [rbp-0x18]
```

Clearly it's moving the into register rdi. We now attempt to print the contents of rdi register:

(gdb) x/9cb $rdi
0x409e10:	98 'b'	50 '2'	32 ' '	98 'b'	111 'o'	109 'm'	98 'b'	101 'e'
0x409e18:	114 'r'

Notice the string here: "b2 bomber". This looks like a passcode! Voila.
The above example is only a mock example. In reality your passcode will be different.

But why did we single out just this instruction above? There are scores of instructions, and we could have singled out any other one. The answer lies in the code you see ahead of that instruction:

```
=> 0x0000000000402dec <+28>:	mov    esi,0x40575d
   0x0000000000402df1 <+33>:	call   0x4010f0 <strcmp@plt>
```

Notice the call to strcmp above. This is usually an indication that it's about the compare a string with another and that would narrow it down to the place in the code where there is a potential hidden clue.

However, calls to strcmp may not be the only indication that passcodes are being compared. But some form of comparison needs to happen in order to compare the passcode entered by the player.

### Info on Registers

The main page can be found here:

```
https://sourceware.org/gdb/current/onlinedocs/gdb/Registers.html#Registers
```

This command is useful to display all the information in the registers:

```
info registers
```

It will display something like:

```
rax            0xb                 11
rbx            0x403b40            4209472
rcx            0x10                16
rdx            0x409df0            4234736
rsi            0x409c10            4234256
rdi            0x409df0            4234736
rbp            0x7fffffffdb40      0x7fffffffdb40
rsp            0x7fffffffdb10      0x7fffffffdb10
r8             0x0                 0
r9             0x7fffffffda30      140737488345648
r10            0xffffffffffffff80  -128
r11            0x202               514
r12            0x401250            4198992
r13            0x7fffffffdca0      140737488346272
r14            0x0                 0
r15            0x0                 0
rip            0x4030cd            0x4030cd <open_door1+93>
eflags         0x246               [ PF ZF IF ]
cs             0x33                51
ss             0x2b                43
ds             0x0                 0
es             0x0                 0
fs             0x0                 0
gs             0x0                 0
k0             0x0                 0
k1             0x0                 0
k2             0x0                 0
k3             0x0                 0
k4             0x0                 0
k5             0x0                 0
k6             0x0                 0
k7             0x0                 0
```
