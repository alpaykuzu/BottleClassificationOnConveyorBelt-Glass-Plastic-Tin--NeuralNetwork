#include "Stepper-28BYJ-48.h"
#include <Servo.h>

Servo myservo1;  
Servo myservo2;

Stepper motor;
bool motorRunning = false;
bool servo1CommandReceived = false;
bool servo2CommandReceived = false;
unsigned long servo1StartTime = 0;
unsigned long servo2StartTime = 0;

void setup() {
  Serial.begin(9600);
  motor.setRPM(40); 
  motor.setAccerelation(400);
  motor.setNewPosition(4076);
  myservo1.attach(2);
  myservo2.attach(3); 
  myservo1.write(180);
  myservo2.write(5); 
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read(); 
    
    if (command == '1') {
      motorRunning = true;
    } else if (command == '0') {
      myservo1.write(180);
      myservo2.write(10);
      motorRunning = false;
    } else if (command == '3') {
      servo2CommandReceived = true;
      servo2StartTime = millis(); 
    } else if (command == '4') {
      servo1CommandReceived = true;
      servo1StartTime = millis(); 
    }
  }

  if (motorRunning) {
    motor.rotateCW(); 
  } else {
    motor.stop(); 
  }

  if (servo1CommandReceived && (millis() - servo1StartTime >= 10700)) { 
    myservo1.write(110); 
    servo1CommandReceived = false; 
  }

  if (servo2CommandReceived && (millis() - servo2StartTime >= 6750)) { 
    myservo2.write(75); 
    servo2CommandReceived = false; 
  }
}
