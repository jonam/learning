The original Ethernet topology was indeed a bus topology, where all devices shared a single communication line, or "bus". This is known as **10BASE5** or **Thicknet**, and it indeed worked like a tap where all nodes were connected to the same coaxial cable. However, there were several challenges and limitations with this approach:

### Challenges with Bus Topology:
1. **Collisions:**
   - Because all devices shared the same communication line, only one device could transmit at a time. If two devices tried to transmit at the same time, a collision would occur, necessitating retransmission and leading to inefficiencies.

2. **Scalability:**
   - The bus topology was not scalable. As more devices were added to the network, the likelihood of collisions increased, leading to degraded network performance.

3. **Distance Limitations:**
   - The maximum length of the coaxial cable and the number of devices that could be connected were limited, restricting the size and expandability of the network.

4. **Maintenance:**
   - Maintenance and fault isolation were problematic. A fault anywhere along the cable often brought down the entire network.

### Evolution to Switched Ethernet:
To overcome these limitations, Ethernet evolved to use a star topology, with Ethernet switches (or hubs, in earlier iterations) at the center, connecting multiple devices.

### Role of a Switch:
1. **Collision Domains:**
   - A switch creates separate collision domains for each connected device, allowing multiple devices to transmit simultaneously without collisions, significantly improving network efficiency and bandwidth utilization compared to a bus topology.
   
2. **Frame Forwarding:**
   - Switches operate at Layer 2 (Data Link Layer) and use MAC addresses to intelligently forward frames only to the port where the destination device is connected, reducing unnecessary traffic on other network segments.
   
3. **Scalability & Flexibility:**
   - Switches allow for the connection of many devices, supporting larger and more scalable networks. They also accommodate easy addition, removal, and relocation of devices.
   
4. **VLANs:**
   - Switches support Virtual Local Area Networks (VLANs) that can segregate traffic within the same physical network, improving security and traffic management.

5. **Full-Duplex Communication:**
   - Modern switches support full-duplex communication, enabling connected devices to transmit and receive simultaneously, effectively doubling the available bandwidth.

### Summary:
While the original Ethernet did use a bus topology, the evolution to a switched Ethernet network using a star topology was driven by the need to overcome the limitations and challenges associated with bus topology. Ethernet switches play a crucial role in modern network architectures by providing intelligent frame forwarding, supporting scalability, reducing collisions, and enabling more efficient and flexible network configurations.
