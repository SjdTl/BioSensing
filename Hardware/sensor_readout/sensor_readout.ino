#include "SerialTransfer.h"
SerialTransfer myTransfer;

struct Data {
  unsigned long Timestamp;
  uint16_t ECG;
  uint16_t GSR;
} testData;

void setup() {
  Serial.begin(57600);
  Serial1.begin(57600);
  myTransfer.begin(Serial1);
  testStruct.Timestamp = 0;
  testStruct.ECG = 0;
  testStruct.GSR = 0;
}

void loop() {
  // use this variable to keep track of how many
  // bytes we’re stuffing in the transmit buffer
  uint16_t sendSize = 0;
  ///////////////////////////////////////// Stuff buffer with struct
  sendSize = myTransfer.txObj(testStruct, sendSize);
  ///////////////////////////////////////// Stuff buffer with array
  sendSize = myTransfer.txObj(arr, sendSize);
  ///////////////////////////////////////// Send buffer
  myTransfer.sendData(sendSize);
  delay(10);
}
