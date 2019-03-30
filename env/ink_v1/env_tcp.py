import numpy as np

import socket
import threading

import json

def env_server(env, tcp_port):
    TCP_PORT = tcp_port
    #Create The Socket
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    #Listen The Port
    s.bind(('',TCP_PORT))
    s.listen(5)
    print( 'TCP_PORT: ', TCP_PORT, " , ", 'Waiting for connection...')

    def tcplink(sock,addr):
        print('Accept new connection from %s:%s...' % addr)
        sock.send( ('Welcome!').encode() )
        while True:
            data=sock.recv(1024).decode()
            # print("data: ", data)

            data_json = json.loads( data )
            # print('data_json["type"]: ', data_json["type"])

            if data_json["type"] == "init":
                data = { 'state_dim' : env.state_dim, 
                'action_dim' : env.action_dim, 
                'DoF' : env.DoF,
                "max_steps": env.max_steps,
                "action_ampl": env.action_ampl,
                "v_lmt": env.v_lmt,
                "time_step": env.time_step } 

            elif data_json["type"] == "reset":
                state = env.reset()
                data = { 'state' : state.tolist() } 

            elif data_json["type"] == "step":
                a = np.array( data_json["action"] )
                state_next, r, done, info = env.step(a)
                data = { 'state' : state_next.tolist(), 'reward' : r, 'done' : done, 'info' : info } 
            elif data_json["type"] == "close":
                break
                
            str_json = json.dumps(data)
            sock.send( str_json.encode() )
        sock.close()
        print('Connection from %s:%s closed.'%addr)
        
    while True:
        # Accept A New Connection
        sock,addr=s.accept()
        
        # Create A New Thread to Deal with The TCP Connection
        t=threading.Thread(target=tcplink(sock,addr))



class env_client():
    def __init__(self, tcp_port):
        self.s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

        self.s.connect(('127.0.0.1',tcp_port))

        print( self.s.recv(1024).decode())

        data = { 'type' : 'init' } 

        str_json = json.dumps(data)
        self.s.send( str_json.encode() )
        str_recv = self.s.recv(1024).decode()
        
        data_json = json.loads( str_recv )
        
        self.state_dim = data_json["state_dim"]
        self.action_dim = data_json["action_dim"]
        self.DoF = data_json["DoF"]
        self.max_steps = data_json["max_steps"]
        self.action_ampl = data_json["action_ampl"]
        self.v_lmt = data_json["v_lmt"]
        self.time_step = data_json["time_step"]

        # s.send('exit')
        # s.close()        
        
    def  reset(self):
        data = { 'type' : 'reset' } 

        str_json = json.dumps(data)
        self.s.send( str_json.encode() )
        str_recv = self.s.recv(1024).decode()
        
        data_json = json.loads( str_recv )
        
        state = np.array( data_json["state"] )
        return state
    def step(self, action):
        
        data = { 'type' : 'step', 'action' : action.tolist() } 

        str_json = json.dumps(data)
        self.s.send( str_json.encode() )
        str_recv = self.s.recv(1024).decode()
        
        data_json = json.loads( str_recv )
        
        state = np.array( data_json["state"] )
        r = data_json["reward"]
        d = data_json["done"]
        info = data_json["info"]
        
        return state, r, d, info
    def close_tcp(self):
        print(" close tcp ... ")
        data = { 'type' : 'close' } 

        str_json = json.dumps(data)
        self.s.send( str_json.encode() )
        self.s.close()        


