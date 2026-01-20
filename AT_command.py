import serial
import time
# Configure the serial connection, chnage the COM port accordingly
ser = serial.Serial('COM10', 115200, timeout=1)
# Wait for the device to initialize
time.sleep(2)
# Send an AT command and wait for a response
def send_at_command(command):
    ser.write((command + '\r\n').encode())
    response = b''
    while True:
        data = ser.read_until()
        response += data
        if b'OK' in data or b'ERROR' in data:
            break
    return response.decode()
# AT Commands usage
response = send_at_command('AT+QNWPREFCFG="mode_pref",NR5G')#Force to 5G SA
print(response)

response = send_at_command('AT+QENG="servingcell"')#Query servingcell
print(response)

response = send_at_command('AT+QCFG="usbnet",2') # Switch to MBIM mode
print(response)

response = send_at_command('AT+CFUN=1,1')#Save and reset module
print(response)

# Close the serial connection
ser.close()
