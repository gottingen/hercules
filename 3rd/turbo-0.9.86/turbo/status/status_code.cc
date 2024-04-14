// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// Created by jeff on 24-1-12.
//
#include "turbo/status/status_code.h"
#include "turbo/status/error.h"

// Register the error code and message
// 0 is reserved for OK
TURBO_REGISTER_ERRNO(turbo::kOk, "OK");
// 1
TURBO_REGISTER_ERRNO(turbo::kEPERM, "Operation not permitted");
//2
TURBO_REGISTER_ERRNO(turbo::kENOENT, "No such file or directory");
//3
TURBO_REGISTER_ERRNO(turbo::kESRCH, "No such process");
//4
TURBO_REGISTER_ERRNO(turbo::kEINTR, "Interrupted system call");
//5
TURBO_REGISTER_ERRNO(turbo::kEIO, "I/O error");
//6
TURBO_REGISTER_ERRNO(turbo::kENXIO, "No such device or address");
//7
TURBO_REGISTER_ERRNO(turbo::kE2BIG, "Argument list too long");
//8
TURBO_REGISTER_ERRNO(turbo::kENOEXEC, "Exec format error");
//9
TURBO_REGISTER_ERRNO(turbo::kEBADF, "Bad file number");
//10
TURBO_REGISTER_ERRNO(turbo::kECHILD, "No child processes");
//11
TURBO_REGISTER_ERRNO(turbo::kEAGAIN, "Try again");
//12
TURBO_REGISTER_ERRNO(turbo::kENOMEM, "Out of memory");
//13
TURBO_REGISTER_ERRNO(turbo::kEACCES, "Permission denied");
//14
TURBO_REGISTER_ERRNO(turbo::kEFAULT, "Bad address");
//15
TURBO_REGISTER_ERRNO(turbo::kENOTBLK, "Block device required");
//16
TURBO_REGISTER_ERRNO(turbo::kEBUSY, "Device or resource busy");
//17
TURBO_REGISTER_ERRNO(turbo::kEEXIST, "File exists");
//18
TURBO_REGISTER_ERRNO(turbo::kEXDEV, "Cross-device link");
//19
TURBO_REGISTER_ERRNO(turbo::kENODEV, "No such device");
//20
TURBO_REGISTER_ERRNO(turbo::kENOTDIR, "Not a directory");
//21
TURBO_REGISTER_ERRNO(turbo::kEISDIR, "Is a directory");
//22
TURBO_REGISTER_ERRNO(turbo::kEINVAL, "Invalid argument");
//23
TURBO_REGISTER_ERRNO(turbo::kENFILE, "File table overflow");
//24
TURBO_REGISTER_ERRNO(turbo::kEMFILE, "Too many open files");
//25
TURBO_REGISTER_ERRNO(turbo::kENOTTY, "Not a typewriter");
//26
TURBO_REGISTER_ERRNO(turbo::kETXTBSY, "Text file busy");
//27
TURBO_REGISTER_ERRNO(turbo::kEFBIG, "File too large");
//28
TURBO_REGISTER_ERRNO(turbo::kENOSPC, "No space left on device");
//29
TURBO_REGISTER_ERRNO(turbo::kESPIPE, "Illegal seek");
//30
TURBO_REGISTER_ERRNO(turbo::kEROFS, "Read-only file system");
//31
TURBO_REGISTER_ERRNO(turbo::kEMLINK, "Too many links");
//32
TURBO_REGISTER_ERRNO(turbo::kEPIPE, "Broken pipe");
//33
TURBO_REGISTER_ERRNO(turbo::kEDOM, "Math argument out of domain of func");
//34
TURBO_REGISTER_ERRNO(turbo::kERANGE, "Math result not representable");
//35
TURBO_REGISTER_ERRNO(turbo::kEDEADLK, "Resource deadlock would occur");
//36
TURBO_REGISTER_ERRNO(turbo::kENAMETOOLONG, "File name too long");
//37
TURBO_REGISTER_ERRNO(turbo::kENOLCK, "No record locks available");
//38
TURBO_REGISTER_ERRNO(turbo::kENOSYS, "Function not implemented");
//39
TURBO_REGISTER_ERRNO(turbo::kENOTEMPTY, "Directory not empty");
//40
TURBO_REGISTER_ERRNO(turbo::kELOOP, "Too many symbolic links encountered");
//42
TURBO_REGISTER_ERRNO(turbo::kENOMSG, "No message of desired type");
//43
TURBO_REGISTER_ERRNO(turbo::kEIDRM, "Identifier removed");
//44
TURBO_REGISTER_ERRNO(turbo::kECHRNG, "Channel number out of range");
//45
TURBO_REGISTER_ERRNO(turbo::kEL2NSYNC, "Level 2 not synchronized");
//46
TURBO_REGISTER_ERRNO(turbo::kEL3HLT, "Level 3 halted");
//47
TURBO_REGISTER_ERRNO(turbo::kEL3RST, "Level 3 reset");
//48
TURBO_REGISTER_ERRNO(turbo::kELNRNG, "Link number out of range");
//49
TURBO_REGISTER_ERRNO(turbo::kEUNATCH, "Protocol driver not attached");
//50
TURBO_REGISTER_ERRNO(turbo::kENOCSI, "No CSI structure available");
//51
TURBO_REGISTER_ERRNO(turbo::kEL2HLT, "Level 2 halted");
//52
TURBO_REGISTER_ERRNO(turbo::kEBADE, "Invalid exchange");
//53
TURBO_REGISTER_ERRNO(turbo::kEBADR, "Invalid request descriptor");
//54
TURBO_REGISTER_ERRNO(turbo::kEXFULL, "Exchange full");
//55
TURBO_REGISTER_ERRNO(turbo::kENOANO, "No anode");
//56
TURBO_REGISTER_ERRNO(turbo::kEBADRQC, "Invalid request code");
//57
TURBO_REGISTER_ERRNO(turbo::kEBADSLT, "Invalid slot");
//59
TURBO_REGISTER_ERRNO(turbo::kEBFONT, "Bad font file format");
//60
TURBO_REGISTER_ERRNO(turbo::kENOSTR, "Device not a stream");
//61
TURBO_REGISTER_ERRNO(turbo::kENODATA, "No data available");
//62
TURBO_REGISTER_ERRNO(turbo::kETIME, "Timer expired");
//63
TURBO_REGISTER_ERRNO(turbo::kENOSR, "Out of streams resources");
//64
TURBO_REGISTER_ERRNO(turbo::kENONET, "Machine is not on the network");
//65
TURBO_REGISTER_ERRNO(turbo::kENOPKG, "Package not installed");
//66
TURBO_REGISTER_ERRNO(turbo::kEREMOTE, "Object is remote");
//67
TURBO_REGISTER_ERRNO(turbo::kENOLINK, "Link has been severed");
//68
TURBO_REGISTER_ERRNO(turbo::kEADV, "Advertise error");
//69
TURBO_REGISTER_ERRNO(turbo::kESRMNT, "Srmount error");
//70
TURBO_REGISTER_ERRNO(turbo::kECOMM, "Communication error on send");
//71
TURBO_REGISTER_ERRNO(turbo::kEPROTO, "Protocol error");
//72
TURBO_REGISTER_ERRNO(turbo::kEMULTIHOP, "Multihop attempted");
//73
TURBO_REGISTER_ERRNO(turbo::kEDOTDOT, "RFS specific error");
//74
TURBO_REGISTER_ERRNO(turbo::kEBADMSG, "Not a data message");
//75
TURBO_REGISTER_ERRNO(turbo::kEOVERFLOW, "Value too large for defined data type");
//76
TURBO_REGISTER_ERRNO(turbo::kENOTUNIQ, "Name not unique on network");
//77
TURBO_REGISTER_ERRNO(turbo::kEBADFD, "File descriptor in bad state");
//78
TURBO_REGISTER_ERRNO(turbo::kEREMCHG, "Remote address changed");
//79
TURBO_REGISTER_ERRNO(turbo::kELIBACC, "Can not access a needed shared library");
//80
TURBO_REGISTER_ERRNO(turbo::kELIBBAD, "Accessing a corrupted shared library");
//81
TURBO_REGISTER_ERRNO(turbo::kELIBSCN, ".lib section in a.out corrupted");
//82
TURBO_REGISTER_ERRNO(turbo::kELIBMAX, "Attempting to link in too many shared libraries");
//83
TURBO_REGISTER_ERRNO(turbo::kELIBEXEC, "Cannot exec a shared library directly");
//84
TURBO_REGISTER_ERRNO(turbo::kEILSEQ, "Illegal byte sequence");
//85
TURBO_REGISTER_ERRNO(turbo::kERESTART, "Interrupted system call should be restarted");
//86
TURBO_REGISTER_ERRNO(turbo::kESTRPIPE, "Streams pipe error");
//87
TURBO_REGISTER_ERRNO(turbo::kEUSERS, "Too many users");
//88
TURBO_REGISTER_ERRNO(turbo::kENOTSOCK, "Socket operation on non-socket");
//89
TURBO_REGISTER_ERRNO(turbo::kEDESTADDRREQ, "Destination address required");
//90
TURBO_REGISTER_ERRNO(turbo::kEMSGSIZE, "Message too long");
//91
TURBO_REGISTER_ERRNO(turbo::kEPROTOTYPE, "Protocol wrong type for socket");
//92
TURBO_REGISTER_ERRNO(turbo::kENOPROTOOPT, "Protocol not available");
//93
TURBO_REGISTER_ERRNO(turbo::kEPROTONOSUPPORT, "Protocol not supported");
//94
TURBO_REGISTER_ERRNO(turbo::kESOCKTNOSUPPORT, "Socket type not supported");
//95
TURBO_REGISTER_ERRNO(turbo::kEOPNOTSUPP, "Operation not supported on transport endpoint");
//96
TURBO_REGISTER_ERRNO(turbo::kEPFNOSUPPORT, "Protocol family not supported");
//97
TURBO_REGISTER_ERRNO(turbo::kEAFNOSUPPORT, "Address family not supported by protocol");
//98
TURBO_REGISTER_ERRNO(turbo::kEADDRINUSE, "Address already in use");
//99
TURBO_REGISTER_ERRNO(turbo::kEADDRNOTAVAIL, "Cannot assign requested address");
//100
TURBO_REGISTER_ERRNO(turbo::kENETDOWN, "Network is down");
//101
TURBO_REGISTER_ERRNO(turbo::kENETUNREACH, "Network is unreachable");
//102
TURBO_REGISTER_ERRNO(turbo::kENETRESET, "Network dropped connection because of reset");
//103
TURBO_REGISTER_ERRNO(turbo::kECONNABORTED, "Software caused connection abort");
//104
TURBO_REGISTER_ERRNO(turbo::kECONNRESET, "Connection reset by peer");
//105
TURBO_REGISTER_ERRNO(turbo::kENOBUFS, "No buffer space available");
//106
TURBO_REGISTER_ERRNO(turbo::kEISCONN, "Transport endpoint is already connected");
//107
TURBO_REGISTER_ERRNO(turbo::kENOTCONN, "Transport endpoint is not connected");
//108
TURBO_REGISTER_ERRNO(turbo::kESHUTDOWN, "Cannot send after transport endpoint shutdown");
//109
TURBO_REGISTER_ERRNO(turbo::kETOOMANYREFS, "Too many references: cannot splice");
//110
TURBO_REGISTER_ERRNO(turbo::kETIMEDOUT, "timed out");
//111
TURBO_REGISTER_ERRNO(turbo::kECONNREFUSED, "Connection refused");
//112
TURBO_REGISTER_ERRNO(turbo::kEHOSTDOWN, "Host is down");
//113
TURBO_REGISTER_ERRNO(turbo::kEHOSTUNREACH, "No route to host");
//114
TURBO_REGISTER_ERRNO(turbo::kEALREADY, "Operation already in progress");
//115
TURBO_REGISTER_ERRNO(turbo::kEINPROGRESS, "Operation now in progress");
//116
TURBO_REGISTER_ERRNO(turbo::kESTALE, "Stale file handle");
//117
TURBO_REGISTER_ERRNO(turbo::kEUCLEAN, "Structure needs cleaning");
//118
TURBO_REGISTER_ERRNO(turbo::kENOTNAM, "Not a XENIX named type file");
//119
TURBO_REGISTER_ERRNO(turbo::kENAVAIL, "No XENIX semaphores available");
//120
TURBO_REGISTER_ERRNO(turbo::kEISNAM, "Is a named type file");
//121
TURBO_REGISTER_ERRNO(turbo::kEREMOTEIO, "Remote I/O error");
//122
TURBO_REGISTER_ERRNO(turbo::kEDQUOT, "Quota exceeded");
//123
TURBO_REGISTER_ERRNO(turbo::kENOMEDIUM, "No medium found");
//124
TURBO_REGISTER_ERRNO(turbo::kEMEDIUMTYPE, "Wrong medium type");
//125
TURBO_REGISTER_ERRNO(turbo::kECANCELED, "Operation Canceled");
//126
TURBO_REGISTER_ERRNO(turbo::kENOKEY, "Required key not available");
//127
TURBO_REGISTER_ERRNO(turbo::kEKEYEXPIRED, "Key has expired");
//128
TURBO_REGISTER_ERRNO(turbo::kEKEYREVOKED, "Key has been revoked");
//129
TURBO_REGISTER_ERRNO(turbo::kEKEYREJECTED, "Key was rejected by service");
//130
TURBO_REGISTER_ERRNO(turbo::kEOWNERDEAD, "Owner died");
//131
TURBO_REGISTER_ERRNO(turbo::kENOTRECOVERABLE, "State not recoverable");
//132
TURBO_REGISTER_ERRNO(turbo::kERFKILL, "Operation not possible due to RF-kill");
//133
TURBO_REGISTER_ERRNO(turbo::kEHWPOISON, "Memory page has hardware error");
//134
TURBO_REGISTER_ERRNO(turbo::kESTOP, "already stop");
//135
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER2, "PLACEHOLDER2");
//136
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER3, "PLACEHOLDER3");
//137
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER4, "PLACEHOLDER4");
//138
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER5, "PLACEHOLDER5");
//139
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER6, "PLACEHOLDER6");
//140
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER7, "PLACEHOLDER7");
//141
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER8, "PLACEHOLDER8");
//142
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER9, "PLACEHOLDER9");
//143
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER10, "PLACEHOLDER10");
//144
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER11, "PLACEHOLDER11");
//145
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER12, "PLACEHOLDER12");
//146
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER13, "PLACEHOLDER13");
//147
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER14, "PLACEHOLDER14");
//148
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER15, "PLACEHOLDER15");
//149
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER16, "PLACEHOLDER16");
//150
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER17, "PLACEHOLDER17");
//151
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER18, "PLACEHOLDER18");
//152
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER19, "PLACEHOLDER19");
//153
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER20, "PLACEHOLDER20");
//154
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER21, "PLACEHOLDER21");
//155
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER22, "PLACEHOLDER22");
//156
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER23, "PLACEHOLDER23");
//157
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER24, "PLACEHOLDER24");
//158
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER25, "PLACEHOLDER25");
//159
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER26, "PLACEHOLDER26");
//160
TURBO_REGISTER_ERRNO(turbo::kEPLACEHOLDER27, "PLACEHOLDER27");
//161
TURBO_REGISTER_ERRNO(turbo::kCancelled, "CANCELLED");
//162
TURBO_REGISTER_ERRNO(turbo::kUnknown, "UNKNOWN");
//163
TURBO_REGISTER_ERRNO(turbo::kInvalidArgument, "INVALID_ARGUMENT");
//164
TURBO_REGISTER_ERRNO(turbo::kDeadlineExceeded, "DEADLINE_EXCEEDED");
//165
TURBO_REGISTER_ERRNO(turbo::kNotFound, "NOT_FOUND");
//166
TURBO_REGISTER_ERRNO(turbo::kAlreadyExists, "ALREADY_EXISTS");
//167
TURBO_REGISTER_ERRNO(turbo::kPermissionDenied, "PERMISSION_DENIED");
//168
TURBO_REGISTER_ERRNO(turbo::kResourceExhausted, "RESOURCE_EXHAUSTED");
//169
TURBO_REGISTER_ERRNO(turbo::kFailedPrecondition, "FAILED_PRECONDITION");
//170
TURBO_REGISTER_ERRNO(turbo::kAborted, "ABORTED");
//171
TURBO_REGISTER_ERRNO(turbo::kOutOfRange, "OUT_OF_RANGE");
//172
TURBO_REGISTER_ERRNO(turbo::kUnimplemented, "UNIMPLEMENTED");
//173
TURBO_REGISTER_ERRNO(turbo::kInternal, "INTERNAL");
//174
TURBO_REGISTER_ERRNO(turbo::kUnavailable, "UNAVAILABLE");
//175
TURBO_REGISTER_ERRNO(turbo::kDataLoss, "DATA_LOSS");
//176
TURBO_REGISTER_ERRNO(turbo::kUnauthenticated, "UNAUTHENTICATED");
//177
TURBO_REGISTER_ERRNO(turbo::kTryAgain, "TRY_AGAIN");
//178
TURBO_REGISTER_ERRNO(turbo::kAlreadyStop, "ALREADY_STOP");
//179
TURBO_REGISTER_ERRNO(turbo::kResourceBusy, "RESOURCE_BUSY");


namespace turbo {

    StatusCode errno_to_status_code(int error_number) {
        if(error_number >= kMinMapStatus && error_number <= kMaxMapStatus){
            return static_cast<StatusCode>(error_number);
        }

        switch (error_number) {
            case 0:
                return kOk;
            case EINVAL:        // Invalid argument
            case ENAMETOOLONG:  // Filename too long
            case E2BIG:         // Argument list too long
            case EDESTADDRREQ:  // Destination address required
            case EDOM:          // Mathematics argument out of domain of function
            case EFAULT:        // Bad address
            case EILSEQ:        // Illegal byte sequence
            case ENOPROTOOPT:   // Protocol not available
            case ENOSTR:        // Not a STREAM
            case ENOTSOCK:      // Not a socket
            case ENOTTY:        // Inappropriate I/O control operation
            case EPROTOTYPE:    // Protocol wrong type for socket
            case ESPIPE:        // Invalid seek
                return kInvalidArgument;
            case ETIMEDOUT:  // Connection timed out
            case ETIME:      // Timer expired
                return kDeadlineExceeded;
            case ENODEV:  // No such device
            case ENOENT:  // No such file or directory
#ifdef ENOMEDIUM
            case ENOMEDIUM:  // No medium found
#endif
            case ENXIO:  // No such device or address
            case ESRCH:  // No such process
                return kNotFound;
            case EEXIST:         // File exists
            case EADDRNOTAVAIL:  // Address not available
            case EALREADY:       // Connection already in progress
#ifdef ENOTUNIQ
            case ENOTUNIQ:  // Name not unique on network
#endif
                return kAlreadyExists;
            case EPERM:   // Operation not permitted
            case EACCES:  // Permission denied
#ifdef ENOKEY
            case ENOKEY:  // Required key not available
#endif
            case EROFS:  // Read only file system
                return kPermissionDenied;
            case ENOTEMPTY:   // Directory not empty
            case EISDIR:      // Is a directory
            case ENOTDIR:     // Not a directory
            case EADDRINUSE:  // Address already in use
            case EBADF:       // Invalid file descriptor
#ifdef EBADFD
            case EBADFD:  // File descriptor in bad state
#endif
            case EBUSY:    // Device or resource busy
            case ECHILD:   // No child processes
            case EISCONN:  // Socket is connected
#ifdef EISNAM
            case EISNAM:  // Is a named type file
#endif
#ifdef ENOTBLK
            case ENOTBLK:  // Block device required
#endif
            case ENOTCONN:  // The socket is not connected
            case EPIPE:     // Broken pipe
#ifdef ESHUTDOWN
            case ESHUTDOWN:  // Cannot send after transport endpoint shutdown
#endif
            case ETXTBSY:  // Text file busy
#ifdef EUNATCH
            case EUNATCH:  // Protocol driver not attached
#endif
                return kFailedPrecondition;
            case ENOSPC:  // No space left on device
#ifdef EDQUOT
            case EDQUOT:  // Disk quota exceeded
#endif
            case EMFILE:   // Too many open files
            case EMLINK:   // Too many links
            case ENFILE:   // Too many open files in system
            case ENOBUFS:  // No buffer space available
            case ENODATA:  // No message is available on the STREAM read queue
            case ENOMEM:   // Not enough space
            case ENOSR:    // No STREAM resources
#ifdef EUSERS
            case EUSERS:  // Too many users
#endif
                return kResourceExhausted;
#ifdef ECHRNG
            case ECHRNG:  // Channel number out of range
#endif
            case EFBIG:      // File too large
            case EOVERFLOW:  // Value too large to be stored in data type
            case ERANGE:     // Result too large
                return kOutOfRange;
#ifdef ENOPKG
            case ENOPKG:  // Package not installed
#endif
            case ENOSYS:        // Function not implemented
            case ENOTSUP:       // Operation not supported
            case EAFNOSUPPORT:  // Address family not supported
#ifdef EPFNOSUPPORT
            case EPFNOSUPPORT:  // Protocol family not supported
#endif
            case EPROTONOSUPPORT:  // Protocol not supported
#ifdef ESOCKTNOSUPPORT
            case ESOCKTNOSUPPORT:  // Socket type not supported
#endif
            case EXDEV:  // Improper link
                return kUnimplemented;
            case EAGAIN:  // Resource temporarily unavailable
#ifdef ECOMM
            case ECOMM:  // Communication error on send
#endif
            case ECONNREFUSED:  // Connection refused
            case ECONNABORTED:  // Connection aborted
            case ECONNRESET:    // Connection reset
            case EINTR:         // Interrupted function call
#ifdef EHOSTDOWN
            case EHOSTDOWN:  // Host is down
#endif
            case EHOSTUNREACH:  // Host is unreachable
            case ENETDOWN:      // Network is down
            case ENETRESET:     // Connection aborted by network
            case ENETUNREACH:   // Network unreachable
            case ENOLCK:        // No locks available
            case ENOLINK:       // Link has been severed
#ifdef ENONET
            case ENONET:  // Machine is not on the network
#endif
                return kUnavailable;
            case EDEADLK:  // Resource deadlock avoided
#ifdef ESTALE
            case ESTALE:  // Stale file handle
#endif
                return kAborted;
            case ECANCELED:  // Operation cancelled
                return kCancelled;
            default:
                return kUnknown;
        }
    }
}
