#include <WinSock2.h>
#include <stdio.h>


int main(){
    WSADATA wsaData; //Create a WSADATA object called wsaData.
    // Initialize Winsock  
    int iResult;
    iResult = WSAStartup(MAKEWORD(2,2), &wsaData); //The WSADATA structure contains information about the Windows Sockets implementation. The MAKEWORD(2,2) parameter of WSAStartup makes a request for version 2.2 of Winsock on the system, and sets the passed version as the highest version of Windows Sockets support that the caller can use.
    if (iResult != 0) {
        printf("WSAStartup failed: %d\n", iResult);
        return 1;
    }   
}