#!/usr/bin/python
"""Custom topology example



Linear Topology,



Switch1-----Switch2------Switch3-----Switch4





Switch1-----Host1

Switch2-----Host2

Switch3-----Host3

Switch4-----Host4



1) pingall

2) bring down link switch3 to Switch4

   h4 will not ping

3) do pingall and check

4) bring up link switch3 to switch4 again

   all should ping

5) do pingall and check





ryu stuff:



ryu-manager ryu.app.simple_switch_13



"""

from mininet.topo import Topo

from mininet.net import Mininet

from mininet.log import setLogLevel, info

from mininet.cli import CLI

from mininet.node import OVSSwitch, Controller, RemoteController

from mininet.node import OVSKernelSwitch

from time import sleep

from mininet.node import CPULimitedHost

from mininet.link import TCLink
from mininet.util import customClass

# Compile and run sFlow helper script
# - configures sFlow on OVS
# - posts topology to sFlow-RT
execfile('sflow-rt/extras/sflow.py')

# Rate limit links to 10Mbps
link = customClass({'tc': TCLink}, 'tc,bw=100')

#from subprocess import call

#from cmd import Cmd

#from os import isatty

#from select import poll, POLLIN

#import sys

#import time

#from mininet.log import info, output, error

#from mininet.term import makeTerms

#from mininet.util import quietRun, isShellBuiltin, dumpNodeConnections


class SingleSwitchTopo(Topo):

    "Single switch connected to n hosts."

    def build(self):

        s1 = self.addSwitch('s1')

        s2 = self.addSwitch('s2')

        s3 = self.addSwitch('s3')

        s4 = self.addSwitch('s4')

        h1 = self.addHost('h1', mac="00:00:00:00:11:01", ip="10.0.0.1/24")

        h2 = self.addHost('h2', mac="00:00:00:00:11:02", ip="10.0.0.2/24")

        h3 = self.addHost('h3', mac="00:00:00:00:11:03", ip="10.0.0.3/24")

        h4 = self.addHost('h4', mac="00:00:00:00:11:04", ip="10.0.0.4/24")

        h5 = self.addHost('h5', mac="00:00:00:00:11:05", ip="10.0.0.5/24")

        h6 = self.addHost('h6', mac="00:00:00:00:11:06", ip="10.0.0.6/24")

        h7 = self.addHost('h7', mac="00:00:00:00:11:07", ip="10.0.0.7/24")

        h8 = self.addHost('h8', mac="00:00:00:00:11:08", ip="10.0.0.8/24")

        h9 = self.addHost('h9', mac="00:00:00:00:11:09", ip="10.0.0.9/24")

        h10 = self.addHost('h10', mac="00:00:00:00:11:10", ip="10.0.0.10/24")

        h11 = self.addHost('h11', mac="00:00:00:00:11:11", ip="10.0.0.11/24")

        h12 = self.addHost('h12', mac="00:00:00:00:11:12", ip="10.0.0.12/24")

        h13 = self.addHost('h13', mac="00:00:00:00:11:13", ip="10.0.0.13/24")

        h14 = self.addHost('h14', mac="00:00:00:00:11:14", ip="10.0.0.14/24")

        h15 = self.addHost('h15', mac="00:00:00:00:11:15", ip="10.0.0.15/24")

        h16 = self.addHost('h16', mac="00:00:00:00:11:16", ip="10.0.0.16/24")

        #h17 = self.addHost('h17', mac="00:00:00:00:11:17", ip="192.168.1.17/24")

        #h18 = self.addHost('h18', mac="00:00:00:00:11:18", ip="192.168.1.18/24")

        #h19 = self.addHost('h19', mac="00:00:00:00:11:19", ip="192.168.1.19/24")

        #h20 = self.addHost('h20', mac="00:00:00:00:11:20", ip="192.168.1.20/24")

        self.addLink(h1, s1)

        self.addLink(h2, s1)

        self.addLink(h3, s1)

        self.addLink(h4, s1)

        self.addLink(h5, s2)

        self.addLink(h6, s2)

        self.addLink(h7, s2)

        self.addLink(h8, s2)

        self.addLink(h9, s3)

        self.addLink(h10, s3)

        self.addLink(h11, s3)

        self.addLink(h12, s3)

        self.addLink(h13, s4)

        self.addLink(h14, s4)

        self.addLink(h15, s4)

        self.addLink(h16, s4, cls=TCLink, bw=10)

        self.addLink(s1, s2)

        self.addLink(s2, s3)

        self.addLink(s3, s4)


if __name__ == '__main__':

    setLogLevel('info')

    topo = SingleSwitchTopo()

    c1 = RemoteController('c1', ip='127.0.0.1')

    net = Mininet(topo=topo, controller=c1, link=link)

    net.start()

    print("TOPOLOGY IS UP")

    #net.iperf((h1, h16))

    #net.pingAll()

    #print("Link S3 to S4 - bringing down - h4 will not be reachable(ping)")

    #net.configLinkStatus('s3', 's4', 'down')

    #print("Link S3 to S4 - bringing up again - all nodes will be reachable")

    #net.configLinkStatus('s3', 's4', 'up')

    ######################################################################

    info(
        ' PHASE 1 - Generate Random Benign Traffic from random IPs to h2 --> h16 hosts\n'
    )

    sleep(5)

    h1 = net.get('h1')

    h2 = net.get('h2')

    h16 = net.get('h16')

    #net.startTerms()

    result = h1.cmd('sudo python Traffic.py -s 2 -e 17 ')

    print 'ARP project', result

    ######################################################################

    info(' PHASE 2 - Generate Attack Traffic from random IPs towards h16\n')

    sleep(5)

    result = h1.cmd('sudo python Attack.py 10.0.0.16')

    print 'ARP project', result

    #######################################################################

    info(
        'PHASE 3 - Simulatenously Generate Random Benign Traffic from random IPs to h2 --> h16 hosts & Attack Traffic random IPs towards h16\n'
    )

    sleep(5)

    h1 = net.get('h1')

    #h2 = net.get('h2')

    # net.startTerms()

    result = h1.cmd('sudo python Traffic.py -s 2 -e 17 &')

    sleep(2)

    #print 'ARP project', result

    #info('Generate Attack Traffic towards h16 ....\n')

    result = h1.cmd('sudo python Attack.py 10.0.0.16 &')
    sleep(20)

    #print    'ARP project', result

    ######################################################################

    CLI(net)

    net.stop()
