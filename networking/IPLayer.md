Routers primarily operate at the Network layer (Layer 3) of the OSI model, which is the IP layer in the TCP/IP stack. The primary function of a router at this layer is to forward packets between different networks based on the destination IP address of the packets. Here’s a brief overview of how routers operate at the IP layer:

### 1. **Routing Table:**
   - Routers maintain a routing table that contains information about known networks and how to reach them. The routing table is populated through manual configuration (static routes) or dynamically through routing protocols.

### 2. **IP Packet Examination:**
   - When a router receives an IP packet, it examines the destination IP address in the packet header to determine the next hop for the packet. The router uses the routing table to make this decision.

### 3. **Packet Forwarding:**
   - Based on the routing table, the router forwards the packet to the next hop, which could be another router or the final destination. The router performs this operation for every packet it receives, regardless of the higher-layer protocol (TCP, UDP, etc.) being used.

### 4. **Decrementing TTL:**
   - As part of processing the IP packet, the router decrements the Time-To-Live (TTL) value in the packet header to prevent packets from looping indefinitely through the network due to routing errors.

### 5. **Layer 2 Handling:**
   - While the routing decisions are made at the Network layer, routers also operate at the Data Link layer (Layer 2) to handle the framing and addressing (e.g., MAC addresses) needed to transmit packets on the connected networks.

### 6. **Interface-specific:**
   - Routers have multiple interfaces, each connected to a different network. Each interface has its own IP address and Layer 2 (Ethernet, for example) address.

### 7. **Network Segmentation:**
   - By operating at the IP layer, routers play a crucial role in segmenting networks and controlling traffic between different network segments, allowing for scalability and organization of network resources.

### 8. **Routing Protocols:**
   - Routers use routing protocols like OSPF, EIGRP, and BGP to exchange routing information with other routers, learn about network topologies, and adapt to changes in the network.

In summary, routers primarily work at the IP (Network) layer to make forwarding decisions and route packets between different networks, but they also interact with other layers of the OSI model to fulfill their roles in the network.
