Writing a program to send a raw Ethernet frame involves using raw sockets, and it indeed requires privileged (root) access, as raw sockets allow the user to construct custom packets, bypassing the protocol stack.

### C Program:
Here’s a simple C program to send a raw Ethernet frame. This program will construct an Ethernet frame with some payload and send it out using a raw socket.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <linux/if_packet.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <Interface Name> <Destination MAC Address>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    const char* ifName = argv[1];
    const char* destMacStr = argv[2];
    
    int sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sockfd == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }
    
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, ifName, IFNAMSIZ - 1);
    
    if (ioctl(sockfd, SIOCGIFINDEX, &ifr) == -1) {
        perror("ioctl");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    uint8_t destMac[6];
    sscanf(destMacStr, "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx", &destMac[0], &destMac[1], &destMac[2], &destMac[3], &destMac[4], &destMac[5]);
    
    struct sockaddr_ll sa;
    memset(&sa, 0, sizeof(struct sockaddr_ll));
    sa.sll_family = AF_PACKET;
    sa.sll_protocol = htons(ETH_P_ALL);
    sa.sll_ifindex = ifr.ifr_ifindex;
    memcpy(sa.sll_addr, destMac, 6);
    sa.sll_halen = 6;
    
    char buffer[64];
    memset(buffer, 'A', 64); // Filling the frame with 'A'
    
    if (sendto(sockfd, buffer, 64, 0, (struct sockaddr*)&sa, sizeof(struct sockaddr_ll)) == -1) {
        perror("sendto");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    printf("Frame sent successfully!\n");
    
    close(sockfd);
    return 0;
}
```

### Explanation:

1. **Choosing the Device:**
   The device (network interface) is chosen based on the user's input (`argv[1]`), and it is typically something like `eth0` for the first Ethernet interface.

2. **Running as Root:**
   This program needs to be run as root (or with sufficient privileges to create raw sockets) since creating raw sockets is a privileged operation.

3. **Guarantee of Frame Being Sent:**
   - The frame is not guaranteed to be sent. Issues can arise at any point; for example, the device could be down, the cable could be unplugged, etc.
   - The `sendto` function returns the number of bytes sent, or `-1` if an error occurred.

4. **Destination MAC Address:**
   - Knowing if the destination MAC address received the frame is challenging at this level, especially if the frame is not an ARP or IP packet.
   - Higher-level protocols (like TCP) include mechanisms for ensuring the delivery of packets and frames, but Ethernet itself does not have a built-in mechanism to acknowledge the receipt of frames.

### Usage:

Compile the C code, and then you can run the program as follows:

```sh
sudo ./a.out eth0 ff:ff:ff:ff:ff:ff
```

Replace `eth0` with the interface you want to use and `ff:ff:ff:ff:ff:ff` with the destination MAC address.
