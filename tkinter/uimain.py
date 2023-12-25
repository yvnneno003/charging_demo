import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont,ImageSequence
import datetime
import time
import configparser
import qrcode
import cv2
import numpy as np
import paho.mqtt.client as mqtt_client
import random
import threading
import json
import calendar
import pytz

serverIp = '60.251.140.228'
port = 1883
client_id = f'python-mqtt-{random.randint(0, 10000)}'
str_message = ""

sub_topic = "device/5760/charger/TW*MSI*E000151"
pub_topic = "user/lilywu@msi.com;ABC-1234"

#global m_plate_num
m_plate_num = "XXX-1111"
#global m_user_id

# 設定字體
font_path = "fonts/arial.ttf"
font_color = "white"

font_size_s = 30
font_s = ImageFont.truetype(font_path, font_size_s)

font_size_s2 = 40
font_s2 = ImageFont.truetype(font_path, font_size_s2)

font_size_b = 50
font_b = ImageFont.truetype(font_path, font_size_b)

#get conf
config = configparser.ConfigParser()
config.read('config/charger.ini')
charger_ID = config.get('DEVICE', 'name')       
print(f'The value of charger_ID is {charger_ID}.')
type = config.get('DEVICE', 'type')       
print(f'The value of type is {type}.')

currency = config.get('DEVICE', 'currency')       
print(f'The value of currency is {currency}.')
charger_fee = config.get('DEVICE', 'charger_fee')       
print(f'The value of charger_fee is {charger_fee}.')
charger_unit = config.get('DEVICE', 'charger_unit')       
print(f'The value of charger_unit is {charger_unit}.')
parking_fee = config.get('DEVICE', 'parking_fee')       
print(f'The value of parking_fee is {parking_fee}.')
parking_unit = config.get('DEVICE', 'parking_unit')       
print(f'The value of parking_unit is {parking_unit}.') 


class MyApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("UI switch")
        self.is_request = False
        self.is_order = False
        self.is_transaction = False
        self.is_qrcode = False
        self.is_Booking = False
        self.all_user_list = []
        self.my_stat=""
        # 创建 MQTT 客户端
        self.mqtt_client = mqtt_client.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message

        # 连接到 MQTT 代理
        self.mqtt_client.connect(serverIp, port, 60)
        self.mqtt_client.loop_start()

        # 创建页面容器
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        # 创建页面字典
        self.frames = {}
        
        # 添加页面到字典
        for F in (Page1, Page2, Page3, Page4, Page5, Page6):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame(Page1) #


     # MQTT 连接成功的回调
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker")
            # 订阅主题
            self.mqtt_client.subscribe(sub_topic)
        else:
            print("Failed to connect to MQTT broker")

    # MQTT 接收消息的回调
    def on_message(self, client, userdata, message):
        print(f"Received message: {message.topic} {message.payload.decode()}")
        str_message = str(message.payload.decode("utf-8"))
        print(f"str_message = `{str_message}`")
        str_split = str_message.split(';')
        print(str_split)
        if len(str_split) == 3 :
            print(str_split[0])#plate_num
            print(str_split[1])#user_id
            print(str_split[2])#sync
            if str_split[2] == 'sync' :
                print('str[2] == sync')
                ls_plate_num = str_split[0]
                ls_user_id = str_split[1]
                pub_topic = "user/"+ls_user_id+";"+ls_plate_num
                print(f"pub_topic='{pub_topic}'")
                # 發布訊息
                msg = "sync success"
                result = client.publish(pub_topic, msg, qos=2)
                status = result[0]
                if status == 0:
                    print(f"Send `{msg}` to topic `{pub_topic}`")
                else:
                    print(f"Failed to send message to topic {pub_topic}")

                self.update_page1(ls_plate_num,ls_user_id)
                self.after(1500, lambda: self.start_sync())
                self.my_stat="occupied"
                self.update_page2(ls_plate_num,ls_user_id)
                self.update_page3(ls_plate_num,ls_user_id)
                self.set_device_state(ls_plate_num,"online", "This device is activated.", "occupied", "This device is occupied.")
                self.send_change()
                print('set_device_state=occupied')    
                self.onRequest()

            elif str_split[2] == 'get_page2_data' :
                print('str[2] == get_page2_data')
                ls_plate_num = str_split[0]
                ls_user_id = str_split[1]
                # 發布訊息
                self.send_page2_data(self.mqtt_client,ls_plate_num,ls_user_id)
            #elif str_split[2] == 'booking' :
            #    print('str[2] == booking')
            #    self.show_frame(Page4)
            elif str_split[2] == 'unlock' :
                print('str[2] == unlock')
                if self.is_Booking == True :
                    self.is_Booking = False
                    ls_user_id = str_split[0]
                    ls_plate_num = str_split[1]
                    print(f"ls_user_id='{ls_user_id}'")
                    print(f"ls_plate_num='{ls_plate_num}'")
                    self.my_stat="occupied"
                    self.set_device_state(ls_plate_num, "online", "This device is activated.", "occupied", "This device is occupied.")
                    self.send_change()
                    print('set_device_state=occupied')
                    self.update_page1(ls_plate_num,ls_user_id)
                    self.show_frame(Page5)
            elif str_split[2] == 'stop_charging' :
                print('str[2] == stop_charging')
                ls_plate_num = str_split[0]
                ls_user_id = str_split[1]
                self.stop_charging()
            elif str_split[2] == 'get_page3_data' :
                print('str[2] == get_page3_data')
                ls_plate_num = str_split[0]
                ls_user_id = str_split[1]
                # 發布訊息
                self.send_page3_data(self.mqtt_client,ls_plate_num,ls_user_id)
            elif str_split[2] == 'action_order' :
                self.is_order = True
                self.action_order()
            elif str_split[0] == 'Login_user' :
                self.add_login_user(str_message)
            else :
                print('str_split[1] <> sync')        
        else :
            print('len(str_split) <> 2')  
            parsed_data = json.loads(str_message)
            print(f"parsed_data = `{parsed_data}`")
            if parsed_data.get("context") == "https://www.msi.com/Result":
                print(f"context = https://www.msi.com/Result")
                if self.is_register==True :
                    print(f"self.is_register==True")
                    self.is_register = False
                    self.my_stat="available"
                    self.set_device_state("device register and online", "online", "This device is activated.", "available", "This device is available.")
                    self.send_change()
                    print('set_device_state=available')
                elif self.is_request==True :
                    print(f"self.is_request==True")
                    if "msg" in parsed_data.get("result", {}):
                        self.is_request = False
                        self.sOrderID=parsed_data["result"]["msg"]
                        print(f"sOrderID = {self.sOrderID}")
                elif self.is_order == True :
                    print(f"self.is_request==True")
                    self.is_order = False
                    ls_result=parsed_data["result"]["return"]
                    print(f"order ls_result = {ls_result}")
                    self.is_transaction = True
                    self.onTransaction()
                elif self.is_transaction == True :
                    print(f"self.is_transaction == True")
                    self.is_transaction = False
                    self.trasactionid=parsed_data["result"]["msg"]
                    print(f"ls_trasactionid = {self.trasactionid}")
                    self.is_qrcode = True
                    self.action_qrcode()
                elif self.is_qrcode == True :
                    print(f"self.is_qrcode == True")
                    try :
                        self.tid=parsed_data["result"]["data"]["trasactionid"]
                        print(f"tid = {self.tid}")
                        self.is_qrcod = False
                        self.qrcode_ok()
                    except KeyError:
                        print(f"json not found transactionid")    
                elif self.is_Booking == True :
                    print(f"self.is_Booking == True")
                    if parsed_data.get("token") =="token-devicestate-get" :
                        
                        data_value = parsed_data["result"]["data"]
                        self.stat=data_value[1].get("state")
                        if self.stat == "available" :
                            self.on_booking_response(self.sBookingID,"accepted", "charger is available", self.m_booking_user+";"+self.m_plate_num)
                            self.my_stat="reserved"
                            self.set_device_state(self.m_booking_user+";"+self.m_plate_num,"online", "This device is activated.", "reserved", "This device has been reserved.")
                            self.send_change()
                            #self.is_Booking = False
                        else:    
                            self.on_booking_response(self.sBookingID, "reject", "charger is not available", self.m_booking_user+";"+self.m_plate_num)
                            self.is_Booking = False        
                else :
                    print(f"else")         
            elif parsed_data.get("context") == "https://www.msi.com/Appointment":
                    if parsed_data.get("contexttype") =="Request" :
                        print(f"parsed_data = {parsed_data}")
                        if self.my_stat == "" or self.my_stat == "available":
                            obj_opt = parsed_data.get("opt") 
                            if obj_opt is not None:
                                self.m_booking_user = obj_opt.get("key01")
                                print("m_booking_user:", self.m_booking_user)
                                self.m_plate_num = obj_opt.get("key02")
                                print("m_plate_num:", self.m_plate_num)
                                if f"{self.m_booking_user};{self.m_plate_num}" not in self.all_user_list:
                                    self.all_user_list.append(f"{self.m_booking_user};{self.m_plate_num}")
                            obj_id = parsed_data.get("id") 
                            if obj_id is not None:
                                self.sBookingID=parsed_data["id"]
                                print(f"self.sBookingID = {self.sBookingID}")
                                self.is_Booking = True
                                self.on_booking_ack(self.sBookingID)
                                self.get_device_state()
                                self.show_frame(Page4)
                            else:
                                print(f"else obj_id is None")
            else:
                print("The 'context' <> 'https://www.msi.com/Result'")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        if page_name==Page1 :
            self.update_page1("","")
            self.update_page2("","")
            self.update_page3("","")
            self.onRegister()
        elif page_name==Page2 :
            self.update_page2_timer()
        elif page_name==Page3 :
            self.update_page3_timer()
        elif page_name==Page4 :
            self.update_page4_timer()
        elif page_name==Page5 :
            self.update_page5_timer()

    def show_page3_data(self, page_name, value1, value2):
        frame = self.frames[page_name]
        frame.tkraise()
        self.update_page3_timer(value1, value2)

    # 自定义方法用于更新页面
    def update_page1(self, plate_num, user_id):
        m_plate_num = plate_num
        print(f'update_page1 m_plate_num: {m_plate_num}')
        m_user_id = user_id
        print(f'update_page1 m_user_id: {m_user_id}')
        page1 = self.frames[Page1]
        page1.update_text(plate_num,user_id)

    def start_sync(self):
        page1 = self.frames[Page1]
        page1.start_sync()    

    # 自定义方法用于更新页面
    def update_page2(self, plate_num, user_id):
        m_plate_num = plate_num
        print(f'update_page2 m_plate_num: {m_plate_num}')
        m_user_id = user_id
        page2 = self.frames[Page2]
        page2.update_text(plate_num,user_id)

    # 自定义方法用于更新页面
    def update_page3(self, plate_num, user_id):
        m_plate_num = plate_num
        print(f'update_page3 m_plate_num: {m_plate_num}')
        m_user_id = user_id
        page3 = self.frames[Page3]
        page3.update_text(plate_num,user_id)        

    def update_page2_timer(self):
        page2 = self.frames[Page2]
        page2.start_timer()
    
    def send_page2_data(self, mqtt_client,ls_plate_num,ls_user_id):
        page2 = self.frames[Page2]
        page2.mqtt_send(mqtt_client,ls_plate_num,ls_user_id)

    def stop_charging(self):
        page2 = self.frames[Page2]
        page2.stop_charging()

    def update_page3_timer(self,value1, value2):
        page3 = self.frames[Page3]
        page3.start_timer()
        page3.update_data(value1, value2)

    def send_page3_data(self, mqtt_client,ls_plate_num,ls_user_id):
        page3 = self.frames[Page3]
        page3.mqtt_send(mqtt_client,ls_plate_num,ls_user_id)

    def action_order(self):
        page3 = self.frames[Page3]
        page3.onOrder(self.sOrderID)

    def action_qrcode(self):
        page3 = self.frames[Page3]
        page3.onQrCode(self.trasactionid)

    def qrcode_ok(self):
        page3 = self.frames[Page3]
        page3.onQrCodeOk()    

    def update_page4_timer(self):
        page4 = self.frames[Page4]
        page4.start_timer()

    def update_page5_timer(self):
        page5 = self.frames[Page5]
        page5.start_timer()    
    
    def onRegister(self):
        utc = int(time.time() * 1000)
        
        obj = {
            "context": "https://www.msi.com/Register",
            "contexttype": "Device",
            "model": "charger",
            "name": "testdevice",
            "project": "5760",
            "serial": charger_ID,
            "token": "1",
            "version": 1,
        }

        sub = {
            "time": utc,
            "unit": "millis",
        }
        obj["registertime"] = sub

        # Use the logging module in Python for equivalent logging
        print("Publish:", json.dumps(obj))
        self.is_register = True
        result = self.mqtt_client.publish('cloud/register', json.dumps(obj).encode(), qos=2)
        status = result[0]
        if status == 0:
            print(f"Send to topic cloud/register")
        else:
            print(f"Failed to send message to topic cloud/register")

    def onRequest(self):
        utc = int(time.time() * 1000)
        
        obj = {
            "context": "https://www.msi.com/Request",
            "version": 1,
            "contexttype": "Order",
            "ordertype": "Create",
            "orderid": "",
            "date": {
                "time": utc,
                "unit": "millis",
            },
            "applicant": {},
            "auth": {
                "context": "https://www.msi.com/Device",
                "version": 1,
                "contexttype": "Auth",
                "project": "5760",
                "model": "charger",
                "serial": charger_ID,
                "token": "",
            },
            "token": "1",
        }

        # Use the logging module in Python for equivalent logging
        msg = json.dumps(obj).encode()
        print("Request Publish:", json.dumps(obj))
        self.is_request = True
        result = self.mqtt_client.publish('cloud/request',msg , qos=2)
        status = result[0]
        if status == 0:
            print(f"Send to topic cloud/request")
        else:
            print(f"Failed to send message to topic cloud/request")

    def onTransaction(self) :
        utc = int(time.time() * 1000)
        obj = {
                "context": "https://www.msi.com/Transaction",
                "version": 1,
                "contexttype": "Order",
                "date": {
                    "time": utc,
                    "unit": "millis"
                },
                "applicant": "",
                "auth": {
                    "context": "https://www.msi.com/Device",
                    "version": 1,
                    "contexttype": "Auth",
                    "project": "5760",
                    "model": "charger",
                    "serial": charger_ID,
                    "token": ""
                },
                "orderid": self.sOrderID,
                "token": "test1"
        }

        print("Transaction: Publish:", json.dumps(obj))
        result = self.mqtt_client.publish('cloud/transaction',json.dumps(obj).encode() , qos=2)
        status = result[0]
        if status == 0:
            print(f"Send to topic cloud/transaction")
        else:
            print(f"Failed to send message to topic cloud/transaction")

    def set_device_state(self,comment1, state1, msg1, state2, msg2):
        try:
            obj = {
                "context": "https://www.msi.com/Devicestate",
                "version": 1,
                "contexttype": "Set",
                "auth": {
                    "context": "https://www.msi.com/Device",
                    "version": 1,
                    "contexttype": "Auth",
                    "project": "5760",
                    "model": "charger",
                    "serial": charger_ID,
                    "token": ""
                },
                "target": {
                    "context": "https://www.msi.com/Device",
                    "version": 1,
                    "contexttype": "Unit",
                    "project": "5760",
                    "model": "charger",
                    "serial": charger_ID
                },
                "code": "1",
                "comment": comment1,
                "states": [
                    {"state": state1, "msg": msg1},
                    {"state": state2, "msg": msg2}
                ],
                "token": "token-devicestate-set"
            }

            print("setDeviceState Complete: Publish: " + json.dumps(obj))
            result = self.mqtt_client.publish("cloud/devicestate", json.dumps(obj).encode() , qos=2)
            status = result[0]
            if status == 0:
                print(f"Send to topic cloud/devicestate")
            else:
                print(f"Failed to send message to topic cloud/devicestate")

        except Exception as e:
            print(e)

    def get_device_state(self):
        try:
            obj = {
                "context": "https://www.msi.com/Devicestate",
                "contexttype": "Get",
                "version": 1,
                "auth": {
                    "context": "https://www.msi.com/Device",
                    "version": 1,
                    "contexttype": "Auth",
                    "project": "5760",
                    "model": "charger",
                    "serial": charger_ID,
                    "token": ""
                },
                "target": {
                    "context": "https://www.msi.com/Device",
                    "version": 1,
                    "contexttype": "Unit",
                    "project": "5760",
                    "model": "charger",
                    "serial": charger_ID
                },
                "token": "token-devicestate-get"
            }

            print("getDeviceState Complete: Publish: " + json.dumps(obj))
            result = self.mqtt_client.publish("cloud/devicestate", json.dumps(obj).encode() , qos=2)
            status = result[0]
            if status == 0:
                print(f"Send getDeviceState to cloud/devicestate")
            else:
                print(f"Failed to send getDeviceState to cloud/devicestate")

        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
        except Exception as e:
            print(e)

    def add_login_user(self,ls_mesg):
        ls_login_user = ls_mesg.replace("Login_user;", "")
        if ls_login_user not in self.all_user_list:
            self.all_user_list.append(ls_login_user)

    def send_change(self):
        print("sendChange(): {}".format(len(self.all_user_list)))
        
        for user in self.all_user_list:
            result = self.mqtt_client.publish("user/{}".format(user), "change")
            status = result[0]
            if status == 0:
                print(f"Send to {user} change")
            else:
                print(f"Failed to send message to {user} change")

    def on_booking_ack(self,m_booking_id):
        try:
            obj = {
                "context": "https://www.msi.com/Appointment",
                "version": 1,
                "contexttype": "Ack",
                "auth": {
                    "context": "https://www.msi.com/Device",
                    "version": 1,
                    "contexttype": "Auth",
                    "project": "5760",
                    "model": "charger",
                    "serial": charger_ID,
                    "token": ""
                },
                "target": {
                    "context": "https://www.msi.com/Device",
                    "version": 1,
                    "contexttype": "Unit",
                    "project": "5760",
                    "model": "charger",
                    "serial": charger_ID
                },
                "uuid": m_booking_id,
                "token": "token-of-ack"
            }

            print("onBookingAck Complete: Publish:", json.dumps(obj))
            result = self.mqtt_client.publish("cloud/appointment", json.dumps(obj).encode() , qos=2)
            status = result[0]
            if status == 0:
                print(f"Send ack to cloud/appointment")
            else:
                print(f"Failed to send ack to cloud/appointment")

        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
        except Exception as e:
            print(e)

    def on_booking_response(self,m_booking_id, ls_return, ls_msg, ls_data):
        try:
            obj = {
                "context": "https://www.msi.com/Appointment",
                "version": 1,
                "contexttype": "Response",
                "auth": {
                    "context": "https://www.msi.com/Device",
                    "version": 1,
                    "contexttype": "Auth",
                    "project": "5760",
                    "model": "charger",
                    "serial": charger_ID,
                    "token": ""
                },
                "target": {
                    "context": "https://www.msi.com/Device",
                    "version": 1,
                    "contexttype": "Unit",
                    "project": "5760",
                    "model": "charger",
                    "serial": charger_ID
                },
                "uuid": m_booking_id,
                "result": [
                    {
                        "return": ls_return,
                        "msg": ls_msg,
                        "data": ls_data,
                        "serial": charger_ID
                    }
                ],
                "token": "token-of-appointment-response"
            }
            print("onBookingResponse Complete: Publish:", json.dumps(obj))
            result = self.mqtt_client.publish("cloud/appointment", json.dumps(obj).encode() , qos=2)
            status = result[0]
            if status == 0:
                print(f"Send Response to cloud/appointment")
            else:
                print(f"Failed to send Response to cloud/appointment")

        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
        except Exception as e:
            print(e)
    
class Page1(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        global tot_per
        tot_per = "70%"
        self.m_plate_num = ""

        w = self.winfo_screenwidth()
        h = self.winfo_screenheight()
        print(f'winfo_screenwidth {w}')
        print(f'winfo_screenheight {h}')

        # 创建背景图像
        self.bg_image = Image.open("img/bg.png")  # 从文件加载背景图像
        self.bg_image = self.bg_image.resize((w, h), Image.ANTIALIAS)  # 调整大小为全屏大小
        self.org_crop = self.bg_image.crop((10,10,310,60))
        bg_photo = ImageTk.PhotoImage(self.bg_image)

        # 创建Label小部件用于显示背景图像
        self.bg_label = tk.Label(self, image=bg_photo)
        self.bg_label.image = bg_photo  # 需要保留对图像的引用，以避免被垃圾回收
        self.bg_label.place(relwidth=1, relheight=1)
        
        def drawUIText(font_text,font_size,font_x,font_y) :
            draw = ImageDraw.Draw(self.bg_image)# 在圖片上繪製文字
            draw.text((font_x, font_y), font_text, font=font_size, fill=font_color)
            photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
            self.bg_label.config(image=photo_image)# 更新Label的圖片
            self.bg_label.image = photo_image

        #Text welcome
        text_w, text_h = font_b.getsize("Welcome")
        x_w = int (w/2 - text_w/2)
        drawUIText("Welcome",font_b,x_w,10)
        
        #Text charger ID
        text_w, text_h = font_s.getsize(charger_ID)
        x_w = w - text_w -20
        drawUIText(charger_ID,font_s,x_w,10)

        #Text Type
        text_w, text_h = font_s.getsize(type)
        x_type = w - text_w - 20
        y_type = 50
        drawUIText(type,font_s,x_type,y_type)

        #Text charger fee
        x_c = int(w * 0.29)
        y_c = int(h * 0.88)
        drawUIText(currency+' '+charger_fee+" / "+charger_unit,font_s,x_c,y_c)

        #Text parking fee
        x_p = int(w * 0.65)
        y_p = int(h * 0.88)
        drawUIText(currency+' '+parking_fee+" / "+parking_unit,font_s,x_p,y_p)

        #image qrcode
        image_q = qrcode.make('charger_id='+charger_ID+",type="+type)
        width_q, height_q = image_q.size
        #print(f'width: {width_q}\nheight: {height_q}')
        x_q = int(w / 2 -  width_q / 2)
        y_q = int(h / 2 - height_q / 2)
        self.bg_image.paste(image_q, (x_q, y_q), mask=None)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image

        #image type
        image_type= Image.open('img/type.png')
        x_type = x_type - 50
        y_type = y_type + 5
        self.bg_image.paste(image_type, (x_type, y_type), mask=None)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image

        #image charge_fee
        image_charge_fee= Image.open('img/e-car.png')
        x_c = x_c - 70
        y_c = y_c - 10
        self.bg_image.paste(image_charge_fee, (x_c, y_c), mask=None)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image

        #image parking_fee
        image_parking_fee= Image.open('img/parking.png')
        x_p = x_p - 50
        y_p = y_p - 10
        self.bg_image.paste(image_parking_fee, (x_p, y_p), mask=None)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image

        time_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        time_label.place(x=10, y=10)

        plate_num_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        plate_num_label.place(x=10, y=60)

        def update_time():
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            # 複製原始位置矩形圖片
            img = self.org_crop.copy()#Image.new('RGBA', (300, 50),(0, 0, 0, 0))
            # 在圖片上繪製文字
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), current_time, font=font_s, fill=font_color)
            # 將圖片轉換成PhotoImage格式
            photo_image = ImageTk.PhotoImage(img)
            # 更新Label的圖片
            time_label.config(image=photo_image)
            time_label.image = photo_image
            self.after(1000, update_time)

        update_time()

        gif = Image.open('img/loading.gif')
        self.img_list = []                                      # 建立儲存影格的空串列
        for frame in ImageSequence.Iterator(gif):
            frame = frame.convert('RGBA')                  # 轉換成 RGBA
            opencv_img = np.array(frame, dtype=np.uint8)   # 轉換成 numpy 陣列
            opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGBA2BGRA)  # 顏色從 RGBA 轉換為 BGRA
            self.img_list.append(opencv_img)                    # 利用串列儲存該圖片資訊
        
        # 子執行緒的工作函數
        def job():
            ######
            loop = True                                        # 設定 loop 為 True
            cv2.namedWindow('Charger',cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Charger',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            start_time = time.time()                     
            while loop == True and time.time() - start_time < 10 :
                for i in self.img_list:          
                    cv2.imshow('Charger', i)                # 不斷讀取並顯示串列中的圖片內容
                    #if cv2.waitKey(10000):
                    if cv2.waitKey(200) == ord('q'):
                        loop = False                           # 停止時同時也將 while 迴圈停止
                        break
            cv2.destroyAllWindows()
            #self.update_page2(m_plate_num,m_user_id)
            self.after(0, lambda: controller.show_frame(Page2))#,self.m_plate_num
            ######

        # 建立一個子執行緒
        #self.t = threading.Thread(target = job)
        
        # 等待 t 這個子執行緒結束
        # t.join()

    # 自定义方法用于更新文本
    def update_text(self, plate_num, user_id):
        # 複製原始位置矩形圖片
        img = self.org_crop.copy()#Image.new('RGBA', (300, 50),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        #draw = ImageDraw.Draw(self.bg_image)# 在圖片上繪製文字
        print(f'plate_num: {plate_num}')
        draw.text((20, 60), plate_num, font=font_s, fill=font_color)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image
        self.m_plate_num = plate_num
        m_user_id = user_id
        print(f'self.m_plate_num: {self.m_plate_num}')

    def start_sync(self) :
        # 執行該子執行緒
        #self.t.start()
        loop = True                                        # 設定 loop 為 True
        cv2.namedWindow('Charger',cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Charger',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        start_time = time.time()                     
        while loop == True and time.time() - start_time < 10 :
            for i in self.img_list:          
                cv2.imshow('Charger', i)                # 不斷讀取並顯示串列中的圖片內容
                #if cv2.waitKey(10000):
                if cv2.waitKey(200) == ord('q'):
                    loop = False                           # 停止時同時也將 while 迴圈停止
                    break
        cv2.destroyAllWindows()
        #self.update_page2(m_plate_num,m_user_id)
        self.after(0, lambda: self.controller.show_frame(Page2))#,self.m_plate_num  

class Page2(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        #self.parameters = {'param1': None, 'param2': None}  # 類別屬性
        self.tot_per = "70%"
        self.loop = False    
        w = self.winfo_screenwidth()
        h = self.winfo_screenheight()
        print(f'winfo_screenwidth {w}')
        print(f'winfo_screenheight {h}')

        # 创建背景图像
        self.bg_image = Image.new('RGB', (w, h),"black")
        bg_photo = ImageTk.PhotoImage(self.bg_image)
        # 创建Label小部件用于显示背景图像
        self.bg_label = tk.Label(self, image=bg_photo)
        self.bg_label.image = bg_photo  # 需要保留对图像的引用，以避免被垃圾回收
        self.bg_label.place(relwidth=1, relheight=1)
        
        def drawUIText(font_text,font_size,font_x,font_y) :
            draw = ImageDraw.Draw(self.bg_image)# 在圖片上繪製文字
            draw.text((font_x, font_y), font_text, font=font_size, fill=font_color)
            photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
            self.bg_label.config(image=photo_image)# 更新Label的圖片
            self.bg_label.image = photo_image

        #Text Charging
        text_w, text_h = font_b.getsize("Charging")
        x_w = int (w/2 - text_w/2)
        drawUIText("Charging",font_b,x_w,10)
        
        #Text charger ID
        text_w, text_h = font_s.getsize(charger_ID)
        x_w = w - text_w -20
        drawUIText(charger_ID,font_s,x_w,10)

        #Text Type
        text_w, text_h = font_s.getsize(type)
        x_type = w - text_w - 20
        y_type = 50
        drawUIText(type,font_s,x_type,y_type)

        #Text plate_num
        x_pn = 20
        y_pn = 60

        self.pn_label = tk.Label(self, borderwidth=0)
        self.pn_label.place(x=x_pn, y=y_pn)
        #drawUIText(m_plate_num,font_s,x_pn,y_pn)

        # 创建一个新图像
        text_w, text_h = font_s.getsize(m_plate_num)
        pn_image = Image.new("RGB", (text_w, text_h), "black")
        # 在新图像上绘制文字
        draw = ImageDraw.Draw(pn_image)
        draw.text((0, 0), m_plate_num, font=font_s, fill=font_color)
        # 将新图像转换为PhotoImage格式
        photo_pn_image = ImageTk.PhotoImage(pn_image)
        # 更新Label的图像
        self.pn_label.config(image=photo_pn_image)
        self.pn_label.image = photo_pn_image


        #battery per
        ls_battery_per = "70%"
        text_w, text_h = font_b.getsize(ls_battery_per)
        x_battery_per = 300
        y_battery_per = 190
        #drawUIText(ls_battery_per,font_b,x_battery_per,y_battery_per)
        self.battery_per_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.battery_per_label.place(x=x_battery_per, y=y_battery_per)

        #Output electricity
        ls_output_e = "Output electricity"
        text_w, text_h = font_s2.getsize(ls_output_e)
        x_output_e = 440
        y_output_e = 190
        drawUIText(ls_output_e,font_s2,x_output_e,y_output_e)

        #Output electricity value
        self.li_output_e_v = 0
        text_w, text_h = font_s2.getsize(str(self.li_output_e_v))
        x_offset = (900 - 750 - text_w) / 2
        x_output_e_v = 750 + x_offset
        y_output_e_v = 190
        #drawUIText(ls_output_e_v,font_s2,x_output_e_v,y_output_e_v)
        self.output_e_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.output_e_label.place(x=x_output_e_v, y=y_output_e_v)

        #Output power value
        ls_output_power_kwh = "kWh"
        text_w, text_h = font_s2.getsize(ls_output_power_kwh)
        x_output_power_kwh = 900
        y_output_power_kwh = 190
        drawUIText(ls_output_power_kwh,font_s2,x_output_power_kwh,y_output_power_kwh)

        #image type
        image_type= Image.open('img/type.png')
        x_type = x_type - 50
        y_type = y_type + 5
        self.bg_image.paste(image_type, (x_type, y_type), mask=None)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image

        # 在背景图像上绘制一条橫线
        draw = ImageDraw.Draw(self.bg_image)
        draw.line((0, 350, w, 350), fill="blue", width=2)
        # 将PIL图像转换为Tkinter PhotoImage
        bg_line = ImageTk.PhotoImage(self.bg_image)
        self.bg_label.config(image=bg_line)# 更新Label的圖片
        self.bg_label.image = bg_line

        # 在背景图像上绘制一条橫线
        draw = ImageDraw.Draw(self.bg_image)
        draw.line((0, 500, w, 500), fill="blue", width=2)
        # 将PIL图像转换为Tkinter PhotoImage
        bg_line = ImageTk.PhotoImage(self.bg_image)
        self.bg_label.config(image=bg_line)# 更新Label的圖片
        self.bg_label.image = bg_line

        # 在背景图像上绘制一条直线
        draw = ImageDraw.Draw(self.bg_image)
        draw.line((360, 350, 360, 500), fill="blue", width=2)
        # 将PIL图像转换为Tkinter PhotoImage
        bg_line = ImageTk.PhotoImage(self.bg_image)
        self.bg_label.config(image=bg_line)# 更新Label的圖片
        self.bg_label.image = bg_line

        # 在背景图像上绘制一条直线
        draw = ImageDraw.Draw(self.bg_image)
        draw.line((682, 350, 682, 500), fill="blue", width=2)
        # 将PIL图像转换为Tkinter PhotoImage
        bg_line = ImageTk.PhotoImage(self.bg_image)
        self.bg_label.config(image=bg_line)# 更新Label的圖片
        self.bg_label.image = bg_line

        #Total charging time
        ls_tot_charge_time = "Total charging time"
        text_w, text_h = font_s2.getsize(ls_tot_charge_time)
        x_tot_charge_time = 10
        y_tot_charge_time = 360
        drawUIText(ls_tot_charge_time,font_s2,x_tot_charge_time,y_tot_charge_time)

        #Total charging time value
        ls_tot_charge_time_v = "00:00:00"
        text_w, text_h = font_s2.getsize(ls_tot_charge_time_v)
        x_offset = (360 - 0 - text_w) / 2
        x_tot_charge_time_v = 0+x_offset
        y_tot_charge_time_v = 430
        #drawUIText(ls_tot_charge_time_v,font_s2,x_tot_charge_time_v,y_tot_charge_time_v)
        self.tot_charge_time_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.tot_charge_time_label.place(x=x_tot_charge_time_v, y=y_tot_charge_time_v)
    
        #time left
        ls_time_left= "Time left"
        text_w, text_h = font_s2.getsize(ls_time_left)
        x_time_left = 430
        y_time_left = 360
        drawUIText(ls_time_left,font_s2,x_time_left,y_time_left)

        #time left value
        self.remaining_time = datetime.timedelta(minutes=3)
        ls_time_left_v= "00:03:00"
        text_w, text_h = font_s2.getsize(ls_time_left_v)
        x_offset = (682 - 360 - text_w) / 2
        x_time_left_v = 360 + x_offset
        y_time_left_v = 430
        #drawUIText(ls_time_left_v,font_s2,x_time_left_v,y_time_left_v)
        self.time_left_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.time_left_label.place(x=x_time_left_v, y=y_time_left_v)

        #Output Power
        ls_output_p = "Output power"
        text_w, text_h = font_s2.getsize(ls_output_p)
        x_output_p = 740
        y_output_p = 360
        drawUIText(ls_output_p,font_s2,x_output_p,y_output_p)

        #Output Power value
        ls_output_p_v = "8410.5 W"
        text_w, text_h = font_s2.getsize(ls_output_p_v)
        x_offset = (1024 - 682 - text_w) / 2
        x_output_p_v = 682 + x_offset
        y_output_p_v = 430
        #drawUIText(ls_output_p_v,font_s2,x_output_p_v,y_output_p_v)
        self.output_power_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.output_power_label.place(x=x_output_p_v, y=y_output_p_v)

        #If you want to terminate, please operate on the mobile APP
        ls_stop_hint = "If you want to terminate, please operate on the mobile APP"
        text_w, text_h = font_s.getsize(ls_stop_hint)
        x_offset = (w - 0 - text_w) / 2
        x_output_p_v = 0 + x_offset
        y_output_p_v = 535
        drawUIText(ls_stop_hint,font_s,x_output_p_v,y_output_p_v)

        #Text time
        self.time_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.time_label.place(x=10, y=10)

        # 拆分动态GIF图像成帧
        gif_frames = []
        gif = Image.open("img/black_universe1.gif")
        try:
            while True:
                gif_frames.append(ImageTk.PhotoImage(gif.copy().resize((200, 200))))#300,225
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        # 创建Label并显示GIF动画
        self.label = tk.Label(self, borderwidth=0)
        self.label.place(x=50, y=120)
        self.play_gif(self.label, gif_frames, 0)

        self.timer1 = None  # 初始化计时器
        self.timer2 = None  # 初始化计时器
        self.send_mqtt_data = False # 初始化计时器
        self.loop = True

    def start_timer(self):
        if self.timer1 is None:
            self.timer1 = self.after(0, self.update_time)
        if self.timer2 is None:
            self.start_time = datetime.datetime.now()
            self.timer2 = self.after(0, self.update_battery_per)    

    def update_time(self):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        # 建立空白圖片
        img = Image.new('RGBA', (300, 50),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), current_time, font=font_s, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.time_label.config(image=photo_image)
        self.time_label.image = photo_image
        self.after(1000, self.update_time)

    def update_battery_per(self):
        #處理輸出功率 w
        ans = random.randint(0, 9)
        # 建立空白圖片
        img = Image.new('RGBA', (300, 50),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), "8410."+str(ans)+" W", font=font_s2, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.output_power_label.config(image=photo_image)
        self.output_power_label.image = photo_image
        ls_output_power="8410."+str(ans)+" W"

        #處理輸出電量 kWh
        #global li_output_e_v  # 使用 global 关键字来告诉 Python 我们要修改全局变量 i
        self.li_output_e_v = self.li_output_e_v + 1
        d = round(self.li_output_e_v / 10.0, 1)
        # 建立空白圖片
        text_w, text_h = font_s2.getsize(str(d))
        img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), str(d), font=font_s2, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.output_e_label.config(image=photo_image)
        self.output_e_label.image = photo_image
        ls_kwh=str(d)

        #電量百分比
        li_kwh = int(d)
        self.tot_per = li_kwh + 70
        if self.tot_per > 100:
            self.tot_per = 100
            self.loop = False
        # 建立空白圖片
        text_w, text_h = font_s2.getsize(str(self.tot_per)+"%")
        img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), str(self.tot_per)+"%", font=font_s2, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.battery_per_label.config(image=photo_image)
        self.battery_per_label.image = photo_image    
        ls_tot_per = str(self.tot_per)+"%"

        #充電時間
        # 计算累积时间
        end_time = datetime.datetime.now()
        elapsed_time = end_time - self.start_time
        # 格式化累积时间为 hh:mm:ss
        #formatted_time = time.elapsed_time.strftime("%H:%M:%S")
        # 手动计算小时、分钟和秒
        seconds = elapsed_time.total_seconds()
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        # 构建格式化字符串
        formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        # 建立空白圖片
        text_w, text_h = font_s2.getsize(formatted_time)
        img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), formatted_time, font=font_s2, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.tot_charge_time_label.config(image=photo_image)
        self.tot_charge_time_label.image = photo_image
        ls_charging_time=formatted_time

        #剩餘時間
        if self.remaining_time > datetime.timedelta(seconds=0) and self.loop:
            self.remaining_time -= datetime.timedelta(seconds=1)
            formatted_time = self.remaining_time - datetime.timedelta(days=self.remaining_time.days)
            formatted_time = formatted_time.total_seconds()
            hours, remainder = divmod(formatted_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            # 建立空白圖片
            text_w, text_h = font_s2.getsize(formatted_time)
            img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
            # 在圖片上繪製文字
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), formatted_time, font=font_s2, fill=font_color)
            # 將圖片轉換成PhotoImage格式
            photo_image = ImageTk.PhotoImage(img)
            # 更新Label的圖片
            self.time_left_label.config(image=photo_image)
            self.time_left_label.image = photo_image
            ls_time_left = formatted_time
            if self.send_mqtt_data == True :
                self.send_data(ls_tot_per,ls_kwh,ls_charging_time,ls_time_left,ls_output_power)
        else:
            self.loop = False
            self.send_mqtt_data = False
            self.stop_charging()
            # 發布訊息
            msg = "start_payment"
            result = self.mqtt_client.publish(pub_topic, msg, qos=2)
            status = result[0]
            if status == 0:
                print(f"Send `{msg}` to topic `{pub_topic}`")
            else:
                print(f"Failed to send message to topic {pub_topic}") 
            self.controller.show_page3_data(Page3,ls_kwh,ls_charging_time)

        if self.loop == True :
            self.after(1000, self.update_battery_per)

    def stop_charging(self) :
        self.loop = False
        print(f"stop_charging loop='{self.loop}'")

    def play_gif(self, label, frames, index):
        label.config(image=frames[index])
        self.after(100, lambda: self.play_gif(label, frames, (index + 1) % len(frames)))
    
    # 自定义方法用于更新文本
    def update_text(self, plate_num, user_id):
        # 创建一个新图像
        text_w, text_h = font_s.getsize(plate_num)
        pn_image = Image.new("RGB", (text_w, text_h), "black")
        # 在新图像上绘制文字
        draw = ImageDraw.Draw(pn_image)
        draw.text((0, 0), plate_num, font=font_s, fill=font_color)
        # 将新图像转换为PhotoImage格式
        photo_pn_image = ImageTk.PhotoImage(pn_image)
        # 更新Label的图像
        self.pn_label.config(image=photo_pn_image)
        self.pn_label.image = photo_pn_image

    def send_data(self,tot_per,kwh,charging_time,time_left,output_power) :
        #pub_topic = "user/"+ls_user_id+";"+ls_plate_num
        pub_topic = "user/"+m_user_id+";"+m_plate_num
        #print(f"pub_topic='{pub_topic}'")
        msg = ""+tot_per+";"+kwh+";"+charging_time+";"+time_left+";"+output_power+";sync update charging"
        #print(f"msg='{msg}'")
        result = self.mqtt_client.publish(pub_topic, msg, qos=2)
        status = result[0]
        #if status == 0:
        #    print(f"Send `{msg}` to topic `{pub_topic}`")
        #else:
        #    print(f"Failed to send message to topic {pub_topic}")  

    def mqtt_send(self, mqtt_client,ls_plate_num,ls_user_id):
        # 發布訊息
        self.send_mqtt_data = True
        global m_plate_num
        global m_user_id
        m_plate_num = ls_plate_num
        m_user_id = ls_user_id
        self.mqtt_client = mqtt_client

class Page3(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        # 添加兩個實例變數來存儲參數
        #self.param1 = None
        #self.param2 = None
        #print(f'self.param1: {self.param1}')
        #print(f'self.param2: {self.param2}')

        w = self.winfo_screenwidth()
        h = self.winfo_screenheight()
        print(f'winfo_screenwidth {w}')
        print(f'winfo_screenheight {h}')

        # 创建背景图像
        self.bg_image = Image.new('RGB', (w, h),"black")
        bg_photo = ImageTk.PhotoImage(self.bg_image)
        # 创建Label小部件用于显示背景图像
        self.bg_label = tk.Label(self, image=bg_photo)
        self.bg_label.image = bg_photo  # 需要保留对图像的引用，以避免被垃圾回收
        self.bg_label.place(relwidth=1, relheight=1)
        
        def drawUIText(font_text,font_size,font_x,font_y) :
            draw = ImageDraw.Draw(self.bg_image)# 在圖片上繪製文字
            draw.text((font_x, font_y), font_text, font=font_size, fill=font_color)
            photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
            self.bg_label.config(image=photo_image)# 更新Label的圖片
            self.bg_label.image = photo_image

        #Text Payment
        text_w, text_h = font_b.getsize("Payment")
        x_w = int (w/2 - text_w/2)
        #drawUIText("Payment",font_b,x_w,10)
        self.payment_label = tk.Label(self, text = "Payment",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.payment_label.place(x=x_w, y=10)
        # 建立空白圖片
        img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), "Payment", font=font_b, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.payment_label.config(image=photo_image)
        self.payment_label.image = photo_image

        #Text charger ID
        text_w, text_h = font_s.getsize(charger_ID)
        x_w = w - text_w -20
        drawUIText(charger_ID,font_s,x_w,10)

        #Text Type
        text_w, text_h = font_s.getsize(type)
        x_type = w - text_w - 20
        y_type = 50
        drawUIText(type,font_s,x_type,y_type)

        #Text plate_num
        x_pn = 20
        y_pn = 60

        self.pn_label = tk.Label(self, borderwidth=0)
        self.pn_label.place(x=x_pn, y=y_pn)
        #drawUIText(m_plate_num,font_s,x_pn,y_pn)

        # 创建一个新图像
        text_w, text_h = font_s.getsize(m_plate_num)
        pn_image = Image.new("RGB", (text_w, text_h), "black")
        # 在新图像上绘制文字
        draw = ImageDraw.Draw(pn_image)
        draw.text((0, 0), m_plate_num, font=font_s, fill=font_color)
        # 将新图像转换为PhotoImage格式
        photo_pn_image = ImageTk.PhotoImage(pn_image)
        # 更新Label的图像
        self.pn_label.config(image=photo_pn_image)
        self.pn_label.image = photo_pn_image

        #Text time
        self.time_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.time_label.place(x=10, y=10)
        self.timer1 = None  # 初始化计时器

         # 在背景图像上绘制一条橫线
        draw = ImageDraw.Draw(self.bg_image)
        draw.line((0, 150, w, 150), fill="blue", width=2)
        # 将PIL图像转换为Tkinter PhotoImage
        bg_line = ImageTk.PhotoImage(self.bg_image)
        self.bg_label.config(image=bg_line)# 更新Label的圖片
        self.bg_label.image = bg_line

        # 在背景图像上绘制一条橫线
        draw = ImageDraw.Draw(self.bg_image)
        draw.line((0, 300, w, 300), fill="blue", width=2)
        # 将PIL图像转换为Tkinter PhotoImage
        bg_line = ImageTk.PhotoImage(self.bg_image)
        self.bg_label.config(image=bg_line)# 更新Label的圖片
        self.bg_label.image = bg_line

        #image type
        image_type= Image.open('img/type.png')
        x_type = x_type - 50
        y_type = y_type + 5
        self.bg_image.paste(image_type, (x_type, y_type), mask=None)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image

        #Charging Fee
        ls_charging_fee = "Charging Fee"
        text_w, text_h = font_s2.getsize(ls_charging_fee)
        x_charging_fee = 100
        y_charging_fee = 160
        drawUIText(ls_charging_fee,font_s2,x_charging_fee,y_charging_fee)

        #Charging Fee value
        ls_charging_fee_v= "TWD  0 "
        text_w, text_h = font_s.getsize(ls_charging_fee_v)
        x_offset = (370 - 80 - text_w) / 2
        x_charging_fee_v = 80 + x_offset
        y_charging_fee_v = 230
        #drawUIText(ls_time_left_v,font_s2,x_time_left_v,y_time_left_v)
        self.charging_fee_label = tk.Label(self, text = ls_charging_fee_v,fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.charging_fee_label.place(x=x_charging_fee_v, y=y_charging_fee_v)
        # 建立空白圖片
        img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), ls_charging_fee_v, font=font_s, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.charging_fee_label.config(image=photo_image)
        self.charging_fee_label.image = photo_image


        #image Charging Fee
        image_charging_fee= Image.open('img/p1.png')
        x_charging_fee = x_charging_fee - 80
        y_charging_fee = y_charging_fee + 35
        self.bg_image.paste(image_charging_fee, (x_charging_fee, y_charging_fee), mask=None)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image

        #Parking Fee
        ls_parking_fee = "Parking Fee"
        text_w, text_h = font_s2.getsize(ls_parking_fee)
        x_parking_fee = 460
        y_parking_fee = 160
        drawUIText(ls_parking_fee,font_s2,x_parking_fee,y_parking_fee)

        #Parking Fee value
        ls_parking_fee_v= "TWD  0 "
        text_w, text_h = font_s.getsize(ls_parking_fee_v)
        x_offset = (710 - 460 - text_w) / 2
        x_parking_fee_v = 460 + x_offset
        y_parking_fee_v = 230
        #drawUIText(ls_time_left_v,font_s2,x_time_left_v,y_time_left_v)
        self.parking_fee_label = tk.Label(self, text = ls_parking_fee_v,fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.parking_fee_label.place(x=x_parking_fee_v, y=y_parking_fee_v)
        # 建立空白圖片
        img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), ls_parking_fee_v, font=font_s, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.parking_fee_label.config(image=photo_image)
        self.parking_fee_label.image = photo_image

        #image Parking Fee
        image_parking_fee= Image.open('img/p2.png')
        x_parking_fee = x_parking_fee - 90
        y_parking_fee = y_parking_fee + 25
        self.bg_image.paste(image_parking_fee, (x_parking_fee, y_parking_fee), mask=None)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image

        #Booking Fee
        ls_booking_fee = "Booking Fee"
        text_w, text_h = font_s2.getsize(ls_booking_fee)
        x_booking_fee = 790
        y_booking_fee = 160
        drawUIText(ls_booking_fee,font_s2,x_booking_fee,y_booking_fee)  

        #Booking Fee value
        ls_booking_fee_v= "TWD  0 "
        text_w, text_h = font_s.getsize(ls_booking_fee_v)
        x_offset = (1024 - 790 - text_w) / 2
        x_booking_fee_v = 790 + x_offset
        y_booking_fee_v = 230
        #drawUIText(ls_time_left_v,font_s2,x_time_left_v,y_time_left_v)
        self.booking_fee_label = tk.Label(self, text = ls_booking_fee_v,fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.booking_fee_label.place(x=x_booking_fee_v, y=y_booking_fee_v)
        # 建立空白圖片
        img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), ls_booking_fee_v, font=font_s, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.booking_fee_label.config(image=photo_image)
        self.booking_fee_label.image = photo_image
        
        #image Booking Fee
        image_booking_fee= Image.open('img/p3.png')
        x_booking_fee = x_booking_fee - 80
        y_booking_fee = y_booking_fee + 30
        self.bg_image.paste(image_booking_fee, (x_booking_fee, y_booking_fee), mask=None)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image

        #Output electricity
        ls_output_e = "Output electricity (TWD 8/kWh) :"
        text_w, text_h = font_s.getsize(ls_output_e)
        x_output_e = 10
        y_output_e = 315
        drawUIText(ls_output_e,font_s,x_output_e,y_output_e)

        #Output electricity value
        li_output_e_v = "0"
        x_output_e_v = 10+text_w+10
        y_output_e_v = 315
        #drawUIText(li_output_e_v,font_s,x_output_e_v,y_output_e_v)
        self.output_e_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.output_e_label.place(x=x_output_e_v, y=y_output_e_v)   

        #Total charging time
        ls_tot_charge_time = "Total charging time :"
        text_w, text_h = font_s.getsize(ls_tot_charge_time)
        x_tot_charge_time = 10
        y_tot_charge_time = 370
        drawUIText(ls_tot_charge_time,font_s,x_tot_charge_time,y_tot_charge_time)

        #Total charging time value
        ls_tot_charge_time_v = "00:00:00"
        #text_w, text_h = font_s.getsize(ls_tot_charge_time_v)
        x_tot_charge_time_v = 10+text_w+10
        y_tot_charge_time_v = 370
        #drawUIText(ls_tot_charge_time_v,font_s,x_tot_charge_time_v,y_tot_charge_time_v)
        self.tot_charge_time_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.tot_charge_time_label.place(x=x_tot_charge_time_v, y=y_tot_charge_time_v)

        #Total parking time
        ls_tot_parking_time = "Total parking time (TWD 40/30min) :"
        text_w, text_h = font_s.getsize(ls_tot_parking_time)
        x_tot_parking_time = 10
        y_tot_parking_time = 425
        drawUIText(ls_tot_parking_time,font_s,x_tot_parking_time,y_tot_parking_time)

        #Total parking time value
        ls_tot_parking_time_v = "00:00:00"
        #text_w, text_h = font_s.getsize(ls_tot_parking_time_v)
        x_tot_parking_time_v = 10+text_w+10
        y_tot_parking_time_v = 415
        #drawUIText(ls_tot_parking_time_v,font_s,x_tot_parking_time_v,y_tot_parking_time_v)
        self.tot_parking_time_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.tot_parking_time_label.place(x=x_tot_parking_time_v, y=y_tot_parking_time_v)

        #Total booking time
        ls_tot_booking_time = "Total booking time :"
        text_w, text_h = font_s.getsize(ls_tot_booking_time)
        x_tot_booking_time = 10
        y_tot_booking_time = 490
        drawUIText(ls_tot_booking_time,font_s,x_tot_booking_time,y_tot_booking_time)

        #Total booking time value
        ls_tot_booking_time_v = "00:00:00"
        #text_w, text_h = font_s.getsize(ls_tot_booking_time_v)
        x_tot_booking_time_v = 10+text_w+10
        y_tot_booking_time_v = 490
        drawUIText(ls_tot_booking_time_v,font_s,x_tot_booking_time_v,y_tot_booking_time_v)
        #self.tot_booking_time_label = tk.Label(self, text = "",fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        #self.tot_booking_time_label.place(x=x_tot_booking_time_v, y=y_tot_booking_time_v)

        #Please pay the fees by your mobile APP
        ls_pay_hint = "Please pay the fees by your mobile APP"
        text_w, text_h = font_s.getsize(ls_pay_hint)
        x_output_p_v = 10
        y_output_p_v = 555
        drawUIText(ls_pay_hint,font_s,x_output_p_v,y_output_p_v)

        self.timer1 = None  # 初始化计时器
        self.charging_rate = 8  # 每kwh的费用
        self.parking_time_list = [0]
        self.extra_intervals = 0
        self.parking_rate = 40  # 每30分钟的费用
        self.parking_free_time =  15 * 60  # 免费停车的时间（15分钟）
        self.send_mqtt_data = False # 初始化计时器
        self.loop = False

    def start_timer(self):
        if self.timer1 is None:
            self.loop = True
            self.timer1 = self.after(0, self.update_time) 

    def format_time(self, hours, minutes, seconds):
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def accumulate_time(self):
        total_seconds = sum(self.parking_time_list)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return hours, minutes, seconds
    
    def parking_calculate_fee(self):
        total_seconds = sum(self.parking_time_list)
        if total_seconds <= self.parking_free_time:
            return 0
        else:
            extra_time = total_seconds - self.parking_free_time
            self.extra_intervals = (extra_time + 1799) // 1800  # 向上取整，每30分钟一个计费周期
            fee = self.extra_intervals * self.parking_rate
            return fee
        
    def update_time(self):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        # 建立空白圖片
        img = Image.new('RGBA', (300, 50),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), current_time, font=font_s, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.time_label.config(image=photo_image)
        self.time_label.image = photo_image

        if self.loop == True :
            #tot_parking_time
            self.parking_time_list[0] += 1  # 每秒增加1秒
            accumulated_hours, accumulated_minutes, accumulated_seconds = self.accumulate_time()
            self.formatted_time = self.format_time(accumulated_hours, accumulated_minutes, accumulated_seconds)
            # 建立空白圖片
            img = Image.new('RGBA', (150, 50),(0, 0, 0, 0))
            # 在圖片上繪製文字
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), self.formatted_time, font=font_s, fill=font_color)
            # 將圖片轉換成PhotoImage格式
            photo_image = ImageTk.PhotoImage(img)
            # 更新Label的圖片
            self.tot_parking_time_label.config(image=photo_image)
            self.tot_parking_time_label.image = photo_image
        
            #self.parking_fee_label
            parking_fee = self.parking_calculate_fee()
            # 建立空白圖片
            text_w, text_h = font_s.getsize('TWD '+str(parking_fee))
            img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
            # 在圖片上繪製文字
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), 'TWD '+str(parking_fee), font=font_s, fill=font_color)
            # 將圖片轉換成PhotoImage格式
            photo_image = ImageTk.PhotoImage(img)
            # 更新Label的圖片
            self.parking_fee_label.config(image=photo_image)
            self.parking_fee_label.image = photo_image
            if self.send_mqtt_data == True :
                self.send_data(str(self.li_kwh),str(parking_fee),"0",self.kwh,self.charger_time,self.formatted_time,"00:00:00","sync update payment")
        
        self.after(1000, self.update_time)

    # 自定义方法用于更新文本
    def update_text(self, plate_num, user_id):
        # 创建一个新图像
        text_w, text_h = font_s.getsize(plate_num)
        pn_image = Image.new("RGB", (text_w, text_h), "black")
        # 在新图像上绘制文字
        draw = ImageDraw.Draw(pn_image)
        draw.text((0, 0), plate_num, font=font_s, fill=font_color)
        # 将新图像转换为PhotoImage格式
        photo_pn_image = ImageTk.PhotoImage(pn_image)
        # 更新Label的图像
        self.pn_label.config(image=photo_pn_image)
        self.pn_label.image = photo_pn_image

    def update_data(self, kwh, charger_time):
        self.kwh=str(kwh)
        self.charger_time=charger_time
        text_w, text_h = font_s.getsize(str(kwh))
        # 建立空白圖片
        img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), kwh, font=font_s, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.output_e_label.config(image=photo_image)
        self.output_e_label.image = photo_image

        text_w, text_h = font_s.getsize(str(charger_time))
        # 建立空白圖片
        img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), charger_time, font=font_s, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.tot_charge_time_label.config(image=photo_image)
        self.tot_charge_time_label.image = photo_image

        #charging fee
        print(f'kwh={kwh}')
        self.li_kwh=int(float(kwh) * self.charging_rate)
        print(f'li_kwh={self.li_kwh}')   
        ls_charging_fee = 'TWD ' +str(self.li_kwh)
        text_w, text_h = font_s.getsize(ls_charging_fee)
        # 建立空白圖片
        img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), ls_charging_fee, font=font_s, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.charging_fee_label.config(image=photo_image)
        self.charging_fee_label.image = photo_image        

        
    def send_data(self,charging_fee,parking_fee,booking_fee,kwh,charger_time,parking_time,booking_time,msg) :
        pub_topic = "user/"+m_user_id+";"+m_plate_num
        #print(f"pub_topic='{pub_topic}'")
        msg = ""+charging_fee+";"+parking_fee+";"+booking_fee+";"+kwh+";"+charger_time+";"+parking_time+";"+booking_time+";"+msg
        #print(f"msg='{msg}'")
        result = self.mqtt_client.publish(pub_topic, msg, qos=2)
        #status = result[0]
        #if status == 0:
        #    print(f"Send `{msg}` to topic `{pub_topic}`")
        #else:
        #    print(f"Failed to send message to topic {pub_topic}")  

    def mqtt_send(self, mqtt_client,ls_plate_num,ls_user_id):
        # 發布訊息
        self.send_mqtt_data = True
        global m_plate_num
        global m_user_id
        m_plate_num = ls_plate_num
        m_user_id = ls_user_id
        self.mqtt_client = mqtt_client

    def onOrder(self,sOrderID) :
        utc = int(time.time() * 1000)

        obj = {}
        sub_date = {}
        sub_applicant = {}
        sub_auth = {}
        sub_arr_orderdata = {}
        sub_arr_orderdata_2 = {}
        sub_arr_orderdata_3 = {}
        sub_arr = []

        obj["context"] = "https://www.msi.com/Order"
        obj["version"] = 1
        obj["contexttype"] = "Update"

        sub_date["time"] = utc 
        sub_date["unit"] = "millis"
        obj["date"] = sub_date

        obj["applicant"] = sub_applicant

        sub_auth["context"] = "https://www.msi.com/Device"
        sub_auth["version"] = 1
        sub_auth["contexttype"] = "Auth"
        sub_auth["project"] = "5760"
        sub_auth["model"] = "charger"
        sub_auth["serial"] = charger_ID
        sub_auth["token"] = ""
        obj["auth"] = sub_auth

        obj["orderid"] = sOrderID

        sub_arr_orderdata["context"] = "https://www.msi.com/Product"
        sub_arr_orderdata["version"] = 1
        sub_arr_orderdata["contexttype"] = "Sell"
        sub_arr_orderdata["id"] = "101"
        sub_arr_orderdata["name"] = "charging fee"
        sub_arr_orderdata["price"] = "8"
        sub_arr_orderdata["quantity"] = self.kwh
        sub_arr_orderdata["unit"] = "TWD"

        sub_arr_orderdata_2["context"] = "https://www.msi.com/Product"
        sub_arr_orderdata_2["version"] = 1
        sub_arr_orderdata_2["contexttype"] = "Sell"
        sub_arr_orderdata_2["id"] = "102"
        sub_arr_orderdata_2["name"] = "parking fee"
        sub_arr_orderdata_2["price"] = "40"
        sub_arr_orderdata_2["quantity"] = str(self.extra_intervals)
        sub_arr_orderdata_2["unit"] = "TWD"

        sub_arr_orderdata_3["context"] = "https://www.msi.com/Product"
        sub_arr_orderdata_3["version"] = 1
        sub_arr_orderdata_3["contexttype"] = "Sell"
        sub_arr_orderdata_3["id"] = "103"
        sub_arr_orderdata_3["name"] = "booking fee"
        sub_arr_orderdata_3["price"] = "0"
        sub_arr_orderdata_3["quantity"] = "0"
        sub_arr_orderdata_3["unit"] = "TWD"

        sub_arr.append(sub_arr_orderdata)
        sub_arr.append(sub_arr_orderdata_2)
        sub_arr.append(sub_arr_orderdata_3)

        obj["orderdata"] = sub_arr

        obj["token"] = "1"

        print("Order: Publish:", json.dumps(obj))
        result = self.mqtt_client.publish('cloud/order', json.dumps(obj).encode(), qos=2)
        status = result[0]
        if status == 0:
            print(f"Send to topic cloud/order")
        else:
            print(f"Failed to send message to topic cloud/order")

    def onQrCode(self,trasactionid) :
        #image qrcode
        image_q = qrcode.make('http://acs-mqtt.ddns.net:8080/payment/mobile/order-check/'+trasactionid)
        image_q_resize = image_q.resize((200,200))
        #width_q, height_q = image_q.size
        x_q = 750
        y_q = 380
        self.bg_image.paste(image_q_resize, (x_q, y_q), mask=None)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image

    def onQrCodeOk(self) :
        self.loop = False
        #image QrCode Ok
        image_qrcode_ok= Image.open('img/p_check.png')
        image_qrcode_ok_resize = image_qrcode_ok.resize((100,100))
        x_qrcode_ok = 800
        y_qrcode_ok = 430
        self.bg_image.paste(image_qrcode_ok_resize, (x_qrcode_ok, y_qrcode_ok), mask=None)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image
        #Completed
        ls_payment_completed= "Completed"
        text_w, text_h = font_b.getsize(ls_payment_completed)
        x_w = int (self.winfo_screenwidth()/2 - text_w/2)
        self.payment_label = tk.Label(self, text = ls_payment_completed,fg="white",bg='#000000', font = ("fixed", 30)) # setting up the labels 
        self.payment_label.place(x=x_w, y=10)
        # 建立空白圖片
        img = Image.new('RGBA', (text_w, text_h),(0, 0, 0, 0))
        # 在圖片上繪製文字
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), ls_payment_completed, font=font_b, fill=font_color)
        # 將圖片轉換成PhotoImage格式
        photo_image = ImageTk.PhotoImage(img)
        # 更新Label的圖片
        self.payment_label.config(image=photo_image)
        self.payment_label.image = photo_image
        self.send_data(str(self.li_kwh),str(parking_fee),"0",self.kwh,self.charger_time,self.formatted_time,"00:00:00","Payment Completed")
        self.after(3000, lambda: self.controller.show_frame(Page1))

class Page4(tk.Frame):#lock
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.tip_count = 5

        w = self.winfo_screenwidth()
        h = self.winfo_screenheight()
        #print(f'winfo_screenwidth {w}')
        #print(f'winfo_screenheight {h}')

        # 创建背景图像
        self.background = cv2.imread('img/bg.png')#Image.open("img/bg.png")  # 从文件加载背景图像
        new_size = (w, h)  # 設定新的寬度和高度
        self.background_resized = cv2.resize(self.background, new_size)
        
        # 設定字體、大小、顏色、粗細等文字屬性
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text = 'Already booked'
        self.font_size = 2
        self.font_thickness = 3
        self.font_color = (0, 0, 255)
        # 取得文字的寬度和高度
        (text_width, text_height), baseline = cv2.getTextSize(self.text, self.font, self.font_size, self.font_thickness)

        # 計算文字在畫面中的位置
        x = (w - text_width) // 2
        y = (h + text_height) // 2

        # 在調整大小後的背景圖上添加文字
        cv2.putText(self.background_resized, self.text,(x, y), self.font, self.font_size, self.font_color,
                    self.font_thickness, cv2.LINE_AA)


        # 將 OpenCV 圖像轉換為 PIL 圖像
        self.background_pil = Image.fromarray(cv2.cvtColor(self.background_resized, cv2.COLOR_BGR2RGB))

        # 將 PIL 圖像轉換為 Tkinter PhotoImage 對象
        self.background_tk = ImageTk.PhotoImage(self.background_pil)

        # 在 Tkinter 視窗上顯示圖像
        self.label = tk.Label(self, image=self.background_tk)
        self.label.pack()
        
        self.base_text = 'Parking space and locked is rising {} s'
        # 保存原始背景圖的副本
        self.original_background_copy = self.background_resized.copy()
        
        self.timer1 = None  # 初始化计时器

    def start_timer(self):
        if self.timer1 is None:
            self.tip_count = 5
            self.timer1 = self.after(0, self.update_time)

    def update_time(self):
        if (self.tip_count > 0) :
            # 在每次更新文字之前，還原為原始背景圖
            self.background_resized = self.original_background_copy.copy()

            new_text = self.base_text.format(self.tip_count)
            
            # 取得文字的寬度和高度
            (text_width, text_height), baseline = cv2.getTextSize(new_text, self.font, 1, 2)

            # 計算文字在畫面中的位置
            x = (self.winfo_screenwidth() - text_width) // 2
            y = self.winfo_screenheight() -10
        
            # 更新 OpenCV 圖像上的文字
            cv2.putText(self.background_resized, new_text, (x, y), self.font, 1,
                                    (255, 255, 255), 2, cv2.LINE_AA)
            # 更新 PIL 圖像
            self.background_pil = Image.fromarray(cv2.cvtColor(self.background_resized, cv2.COLOR_BGR2RGB))
            self.background_tk = ImageTk.PhotoImage(self.background_pil)
            # 更新 Tkinter 視窗上的圖像
            self.label.config(image=self.background_tk)
            # 每秒後再次調用更新文字的方法
            self.tip_count -= 1
            self.after(1000, self.update_time)
        else :
            # 更新 PIL 圖像
            self.background_pil = Image.fromarray(cv2.cvtColor(self.original_background_copy.copy(), cv2.COLOR_BGR2RGB))
            self.background_tk = ImageTk.PhotoImage(self.background_pil)
            # 更新 Tkinter 視窗上的圖像
            self.label.config(image=self.background_tk)
            self.tip_count -= 1
        
class Page5(tk.Frame):#unlock
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.tip_count = 5
        self.controller = controller

        w = self.winfo_screenwidth()
        h = self.winfo_screenheight()
        # 创建背景图像
        self.background = cv2.imread('img/msi.jpg')# 从文件加载背景图像
        new_size = (w, h)  # 設定新的寬度和高度
        self.background_resized = cv2.resize(self.background, new_size)

        # 將 OpenCV 圖像轉換為 PIL 圖像
        self.background_pil = Image.fromarray(cv2.cvtColor(self.background_resized, cv2.COLOR_BGR2RGB))

        # 將 PIL 圖像轉換為 Tkinter PhotoImage 對象
        self.background_tk = ImageTk.PhotoImage(self.background_pil)

        # 在 Tkinter 視窗上顯示圖像
        self.label = tk.Label(self, image=self.background_tk)
        self.label.pack()
        
        self.base_text = 'Parking space and locked is falling {} s'
        # 保存原始背景圖的副本
        self.original_background_copy = self.background_resized.copy()
        
        #self.timer1 = None  # 初始化计时器

    def start_timer(self):
        #if self.timer1 is None:
        self.tip_count = 5
        self.timer1 = self.after(0, self.update_time)

    def update_time(self):
        if (self.tip_count > 0) :
            # 在每次更新文字之前，還原為原始背景圖
            self.background_resized = self.original_background_copy.copy()

            new_text = self.base_text.format(self.tip_count)
            
            # 取得文字的寬度和高度
            (text_width, text_height), baseline = cv2.getTextSize(new_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            # 計算文字在畫面中的位置
            x = (self.winfo_screenwidth() - text_width) // 2
            y = self.winfo_screenheight() -10
        
            # 更新 OpenCV 圖像上的文字
            cv2.putText(self.background_resized, new_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 0), 2, cv2.LINE_AA)
            # 更新 PIL 圖像
            self.background_pil = Image.fromarray(cv2.cvtColor(self.background_resized, cv2.COLOR_BGR2RGB))
            self.background_tk = ImageTk.PhotoImage(self.background_pil)
            # 更新 Tkinter 視窗上的圖像
            self.label.config(image=self.background_tk)
            # 每秒後再次調用更新文字的方法
            self.tip_count -= 1
            self.after(1000, self.update_time)
        else :
            self.tip_count -= 1
            # 切換到Page1
            self.controller.show_frame(Page1)

class Page6(tk.Frame):        
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        w = self.winfo_screenwidth()
        h = self.winfo_screenheight()
        # 创建背景图像
        self.bg_image = Image.open("img/bg.png")  # 从文件加载背景图像
        self.bg_image = self.bg_image.resize((w, h), Image.ANTIALIAS)  # 调整大小为全屏大小
        bg_photo = ImageTk.PhotoImage(self.bg_image)

        # 创建Label小部件用于显示背景图像
        self.bg_label = tk.Label(self, image=bg_photo)
        self.bg_label.image = bg_photo  # 需要保留对图像的引用，以避免被垃圾回收
        self.bg_label.place(relwidth=1, relheight=1)
        
        def drawUIText(font_text,font_size,font_x,font_y) :
            draw = ImageDraw.Draw(self.bg_image)# 在圖片上繪製文字
            draw.text((font_x, font_y), font_text, font=font_size, fill=font_color)
            photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
            self.bg_label.config(image=photo_image)# 更新Label的圖片
            self.bg_label.image = photo_image


        #image qrcode
        image_q = qrcode.make('charger_id='+charger_ID+",type="+type)
        width_q, height_q = image_q.size
        #print(f'width: {width_q}\nheight: {height_q}')
        x_q = int(w / 2 -  width_q / 2)
        y_q = int(h / 2 - height_q / 2)
        self.bg_image.paste(image_q, (x_q, y_q), mask=None)
        photo_image = ImageTk.PhotoImage(self.bg_image)# 將圖片轉換成PhotoImage格式
        self.bg_label.config(image=photo_image)# 更新Label的圖片
        self.bg_label.image = photo_image

        #mesg
        text_mesg = "Please use the mobile app to scan QRCode"
        text_w, text_h = font_s.getsize(text_mesg)
        x_w = int (w/2 - text_w/2)
        y_h = h - text_h - 10
        drawUIText(text_mesg,font_s,x_w,y_h)

if __name__ == "__main__":
    app = MyApp()
    app.attributes('-fullscreen', True)
    app.mainloop()




