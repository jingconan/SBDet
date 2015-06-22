Detection of Botnet using Social Network Analysis
--------------------------------------------------

Modules in SBDet

  - Data.py: this modules defines the file type that can be handled.
    Since the new dataset has a new flow format, you need to add a
    new class that is the subclass PreloadHardDiskFile. This module is
    similar to the Data module in SADIT.
  - SIG.py: this module contains the functions to calculate Social
    Interaction Graphs (SIGs) from raw data. 
  - Models.py: this module contains functions that are used to determine
    the type of random graph model that should be used. The most
    important function is:

        - **sample**: used to sample degrees from a undirected graph.
        - **select_model**: select models based on the samples.
        
  - Monitor.py: this module implements the graph-based anomaly detection
    algorithm. The most important function is:  
	    
	    - **monitor_deg_dis**
	    
  - Community.py: this file contains the community detection related
    functions.

        - **detect_botnet**: this is the main function used to detect
          botnets. This function calls the following questions

            - ident_pivot_nodes: identify pivot nodes.
            - cal_cor_graph: calculate the correlation graph. We will apply
              community detection algorithm to this graph.
            - cal_inta_pnodes: calculate the interaction of nodes to pivot
              nodes. This will be another input of out modified community
              detection algorithm.

Steps to run the algorithm on a dataset.
1. Create social interaction graph from raw data. For pcap files, I created a tool pcap2sigs (https://github.com/hbhzwj/pcap2sigs) that can be used to create sigs output. For the binetflow format, you need to write a tool to create SIG.
2. You can use parseToCoo to parse the output of pcap2sigs.
3. call monitor_deg_dis() to calculate the divergence of each SIGs.
4. Apply the threshold to the divergences, and identify a set of suspicious SIGs.


Installation:
-----------------------------
This tool depends on 

1. numpy
2. matplotlib
2. networkx
3. igraph
4. cython
5. CSDP
Please install numpy, networkx, cython using apt-get. Run the following command (I have tested in Ubuntu 12.04)

```bash
 $ sudo apt-get install python-numpy python-matplotlib python-networkx python-igraph cython
```
For CSDP, there is a binary in ./csdp6.1.0linuxp4 folder, you should
be able to run it directly in Linux. Please add the folder csdp6.1.0linuxp4 to your PATH environment variable.
Run the following command:

```bash
$ export PATH=$PATH:<path/to/csdp6.1.0linuxp4/bin>
```
<path/to/csdp6.1.0linuxp4/bin> is the absolute path of the folder that contains the csdp binary. For example, it is /home/wangjing/Dropbox/Researc
h/CyberSecurity/CommunityDetection/social-bot-detection-git/csdp6.1.0linuxp4/bin in my machine.

In case that the check_call function fails and it complains about
permission problem, please run
    
```bash
 $ chomd +x path/to/csdp6.1.0linuxp4/bin/csdp
```

Demo
----------------
There is a small demo in folder Example/demo. It contains two files.

1. BotDiscCaida.py: the file that generate test data and run SBDet algorithm.
2. PlotDiscSenCaida.py: the file that plots the result.

Run the demo with the following commands:

```bash
$ cd Example/demo
$ python BotDiscCaida.py
$ python PlotDiscSenCaida.py
```


The demo is based on the code I used to generated the TCNS journal paper. The raw code for TCNS journal is in folder Example/social-bot-paper-sim/
