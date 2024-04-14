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

#ifndef TURBO_STATUS_STATUS_CODE_H_
#define TURBO_STATUS_STATUS_CODE_H_

#include <cstdint>
#include <cerrno>
#include <system_error>
#include "turbo/status/error.h"

namespace turbo {
    /// pre register status code for system errno
    // StatusCode::kOk
    //
    // kOK does not indicate an error; this value is returned on
    // success. It is typical to check for this value before proceeding on any
    // given call across an API or RPC boundary. To check this value, use the
    // `turbo::Status::ok()` member function rather than inspecting the raw code.
    TURBO_DECLARE_ERRNO(kOk, 0);
    //static constexpr StatusCode kOk = 0;

    // 1
    TURBO_DECLARE_ERRNO(kEPERM, EPERM);
    //static constexpr StatusCode kEPERM = EPERM;

    // 2
    TURBO_DECLARE_ERRNO(kENOENT, ENOENT);
    //static constexpr StatusCode kENOENT = ENOENT;

    // 3
    TURBO_DECLARE_ERRNO(kESRCH, ESRCH);
    //static constexpr StatusCode kESRCH = ESRCH;

    // 4
    TURBO_DECLARE_ERRNO(kEINTR, EINTR);
    //static constexpr StatusCode kEINTR = EINTR;

    // 5
    TURBO_DECLARE_ERRNO(kEIO, EIO);
    // 6
    TURBO_DECLARE_ERRNO(kENXIO, ENXIO);
    // 7
    TURBO_DECLARE_ERRNO(kE2BIG, E2BIG);
    // 8
    TURBO_DECLARE_ERRNO(kENOEXEC, ENOEXEC);
    // 9
    TURBO_DECLARE_ERRNO(kEBADF, EBADF);
    // 10
    TURBO_DECLARE_ERRNO(kECHILD, ECHILD);
    // 11
    TURBO_DECLARE_ERRNO(kEAGAIN, EAGAIN);
    // 12
    TURBO_DECLARE_ERRNO(kENOMEM, ENOMEM);
    // 13
    TURBO_DECLARE_ERRNO(kEACCES, EACCES);
    // 14
    TURBO_DECLARE_ERRNO(kEFAULT, EFAULT);
    // 15
    TURBO_DECLARE_ERRNO(kENOTBLK, ENOTBLK);
    // 16
    TURBO_DECLARE_ERRNO(kEBUSY, EBUSY);
    // 17
    TURBO_DECLARE_ERRNO(kEEXIST, EEXIST);
    // 18
    TURBO_DECLARE_ERRNO(kEXDEV, EXDEV);
    // 19
    TURBO_DECLARE_ERRNO(kENODEV, ENODEV);
    // 20
    TURBO_DECLARE_ERRNO(kENOTDIR, ENOTDIR);
    // 21
    TURBO_DECLARE_ERRNO(kEISDIR, EISDIR);
    // 22
    TURBO_DECLARE_ERRNO(kEINVAL, EINVAL);
    // 23
    TURBO_DECLARE_ERRNO(kENFILE, ENFILE);
    // 24
    TURBO_DECLARE_ERRNO(kEMFILE, EMFILE);
    // 25
    TURBO_DECLARE_ERRNO(kENOTTY, ENOTTY);
    // 26
    TURBO_DECLARE_ERRNO(kETXTBSY, ETXTBSY);
    // 27
    TURBO_DECLARE_ERRNO(kEFBIG, EFBIG);
    // 28
    TURBO_DECLARE_ERRNO(kENOSPC, ENOSPC);
    // 29
    TURBO_DECLARE_ERRNO(kESPIPE, ESPIPE);
    // 30
    TURBO_DECLARE_ERRNO(kEROFS, EROFS);
    // 31
    TURBO_DECLARE_ERRNO(kEMLINK, EMLINK);
    // 32
    TURBO_DECLARE_ERRNO(kEPIPE, EPIPE);
    // 33
    TURBO_DECLARE_ERRNO(kEDOM, EDOM);
    // 34
    TURBO_DECLARE_ERRNO(kERANGE, ERANGE);
    // 35
    TURBO_DECLARE_ERRNO(kEDEADLK, EDEADLK);
    // 36
    TURBO_DECLARE_ERRNO(kENAMETOOLONG, ENAMETOOLONG);
    // 37
    TURBO_DECLARE_ERRNO(kENOLCK, ENOLCK);
    // 38
    TURBO_DECLARE_ERRNO(kENOSYS, ENOSYS);
    // 39
    TURBO_DECLARE_ERRNO(kENOTEMPTY, ENOTEMPTY);
    // 40
    TURBO_DECLARE_ERRNO(kELOOP, ELOOP);
    // no 41  EAGAIN
    //TURBO_DECLARE_ERRNO(kEWOULDBLOCK, EWOULDBLOCK);
    // 42
    TURBO_DECLARE_ERRNO(kENOMSG, ENOMSG);
    // 43
    TURBO_DECLARE_ERRNO(kEIDRM, EIDRM);
    // 44
    TURBO_DECLARE_ERRNO(kECHRNG, ECHRNG);
    // 45
    TURBO_DECLARE_ERRNO(kEL2NSYNC, EL2NSYNC);
    // 46
    TURBO_DECLARE_ERRNO(kEL3HLT, EL3HLT);
    // 47
    TURBO_DECLARE_ERRNO(kEL3RST, EL3RST);
    // 48
    TURBO_DECLARE_ERRNO(kELNRNG, ELNRNG);
    // 49
    TURBO_DECLARE_ERRNO(kEUNATCH, EUNATCH);
    // 50
    TURBO_DECLARE_ERRNO(kENOCSI, ENOCSI);
    // 51
    TURBO_DECLARE_ERRNO(kEL2HLT, EL2HLT);
    // 52
    TURBO_DECLARE_ERRNO(kEBADE, EBADE);
    // 53
    TURBO_DECLARE_ERRNO(kEBADR, EBADR);
    // 54
    TURBO_DECLARE_ERRNO(kEXFULL, EXFULL);
    // 55
    TURBO_DECLARE_ERRNO(kENOANO, ENOANO);
    // 56
    TURBO_DECLARE_ERRNO(kEBADRQC, EBADRQC);
    // 57
    TURBO_DECLARE_ERRNO(kEBADSLT, EBADSLT);
    // no 58 EDEADLK
    //TURBO_DECLARE_ERRNO(kEDEADLOCK, EDEADLOCK);
    // 59
    TURBO_DECLARE_ERRNO(kEBFONT, EBFONT);
    // 60
    TURBO_DECLARE_ERRNO(kENOSTR, ENOSTR);
    // 61
    TURBO_DECLARE_ERRNO(kENODATA, ENODATA);
    // 62
    TURBO_DECLARE_ERRNO(kETIME, ETIME);
    // 63
    TURBO_DECLARE_ERRNO(kENOSR, ENOSR);
    // 64
    TURBO_DECLARE_ERRNO(kENONET, ENONET);
    // 65
    TURBO_DECLARE_ERRNO(kENOPKG, ENOPKG);
    // 66
    TURBO_DECLARE_ERRNO(kEREMOTE, EREMOTE);
    // 67
    TURBO_DECLARE_ERRNO(kENOLINK, ENOLINK);
    // 68
    TURBO_DECLARE_ERRNO(kEADV, EADV);
    // 69
    TURBO_DECLARE_ERRNO(kESRMNT, ESRMNT);
    // 70
    TURBO_DECLARE_ERRNO(kECOMM, ECOMM);
    // 71
    TURBO_DECLARE_ERRNO(kEPROTO, EPROTO);
    // 72
    TURBO_DECLARE_ERRNO(kEMULTIHOP, EMULTIHOP);
    // 73
    TURBO_DECLARE_ERRNO(kEDOTDOT, EDOTDOT);
    // 74
    TURBO_DECLARE_ERRNO(kEBADMSG, EBADMSG);
    // 75
    TURBO_DECLARE_ERRNO(kEOVERFLOW, EOVERFLOW);
    // 76
    TURBO_DECLARE_ERRNO(kENOTUNIQ, ENOTUNIQ);
    // 77
    TURBO_DECLARE_ERRNO(kEBADFD, EBADFD);
    // 78
    TURBO_DECLARE_ERRNO(kEREMCHG, EREMCHG);
    // 79
    TURBO_DECLARE_ERRNO(kELIBACC, ELIBACC);
    // 80
    TURBO_DECLARE_ERRNO(kELIBBAD, ELIBBAD);
    // 81
    TURBO_DECLARE_ERRNO(kELIBSCN, ELIBSCN);
    // 82
    TURBO_DECLARE_ERRNO(kELIBMAX, ELIBMAX);
    // 83
    TURBO_DECLARE_ERRNO(kELIBEXEC, ELIBEXEC);
    // 84
    TURBO_DECLARE_ERRNO(kEILSEQ, EILSEQ);
    // 85
    TURBO_DECLARE_ERRNO(kERESTART, ERESTART);
    // 86
    TURBO_DECLARE_ERRNO(kESTRPIPE, ESTRPIPE);
    // 87
    TURBO_DECLARE_ERRNO(kEUSERS, EUSERS);
    // 88
    TURBO_DECLARE_ERRNO(kENOTSOCK, ENOTSOCK);
    // 89
    TURBO_DECLARE_ERRNO(kEDESTADDRREQ, EDESTADDRREQ);
    // 90
    TURBO_DECLARE_ERRNO(kEMSGSIZE, EMSGSIZE);
    // 91
    TURBO_DECLARE_ERRNO(kEPROTOTYPE, EPROTOTYPE);
    // 92
    TURBO_DECLARE_ERRNO(kENOPROTOOPT, ENOPROTOOPT);
    // 93
    TURBO_DECLARE_ERRNO(kEPROTONOSUPPORT, EPROTONOSUPPORT);
    // 94
    TURBO_DECLARE_ERRNO(kESOCKTNOSUPPORT, ESOCKTNOSUPPORT);
    // 95
    TURBO_DECLARE_ERRNO(kEOPNOTSUPP, EOPNOTSUPP);
    // 96
    TURBO_DECLARE_ERRNO(kEPFNOSUPPORT, EPFNOSUPPORT);
    // 97
    TURBO_DECLARE_ERRNO(kEAFNOSUPPORT, EAFNOSUPPORT);
    // 98
    TURBO_DECLARE_ERRNO(kEADDRINUSE, EADDRINUSE);
    // 99
    TURBO_DECLARE_ERRNO(kEADDRNOTAVAIL, EADDRNOTAVAIL);
    // 100
    TURBO_DECLARE_ERRNO(kENETDOWN, ENETDOWN);
    // 101
    TURBO_DECLARE_ERRNO(kENETUNREACH, ENETUNREACH);
    // 102
    TURBO_DECLARE_ERRNO(kENETRESET, ENETRESET);
    // 103
    TURBO_DECLARE_ERRNO(kECONNABORTED, ECONNABORTED);
    // 104
    TURBO_DECLARE_ERRNO(kECONNRESET, ECONNRESET);
    // 105
    TURBO_DECLARE_ERRNO(kENOBUFS, ENOBUFS);
    // 106
    TURBO_DECLARE_ERRNO(kEISCONN, EISCONN);
    // 107
    TURBO_DECLARE_ERRNO(kENOTCONN, ENOTCONN);
    // 108
    TURBO_DECLARE_ERRNO(kESHUTDOWN, ESHUTDOWN);
    // 109
    TURBO_DECLARE_ERRNO(kETOOMANYREFS, ETOOMANYREFS);
    // 110
    TURBO_DECLARE_ERRNO(kETIMEDOUT, ETIMEDOUT);
    // 111
    TURBO_DECLARE_ERRNO(kECONNREFUSED, ECONNREFUSED);
    // 112
    TURBO_DECLARE_ERRNO(kEHOSTDOWN, EHOSTDOWN);
    // 113
    TURBO_DECLARE_ERRNO(kEHOSTUNREACH, EHOSTUNREACH);
    // 114
    TURBO_DECLARE_ERRNO(kEALREADY, EALREADY);
    // 115
    TURBO_DECLARE_ERRNO(kEINPROGRESS, EINPROGRESS);
    // 116
    TURBO_DECLARE_ERRNO(kESTALE, ESTALE);
    // 117
    TURBO_DECLARE_ERRNO(kEUCLEAN, EUCLEAN);
    // 118
    TURBO_DECLARE_ERRNO(kENOTNAM, ENOTNAM);
    // 119
    TURBO_DECLARE_ERRNO(kENAVAIL, ENAVAIL);
    // 120
    TURBO_DECLARE_ERRNO(kEISNAM, EISNAM);
    // 121
    TURBO_DECLARE_ERRNO(kEREMOTEIO, EREMOTEIO);
    // 122
    TURBO_DECLARE_ERRNO(kEDQUOT, EDQUOT);
    // 123
    TURBO_DECLARE_ERRNO(kENOMEDIUM, ENOMEDIUM);
    // 124
    TURBO_DECLARE_ERRNO(kEMEDIUMTYPE, EMEDIUMTYPE);
    // 125
    TURBO_DECLARE_ERRNO(kECANCELED, ECANCELED);
    // 126
    TURBO_DECLARE_ERRNO(kENOKEY, ENOKEY);
    // 127
    TURBO_DECLARE_ERRNO(kEKEYEXPIRED, EKEYEXPIRED);
    // 128
    TURBO_DECLARE_ERRNO(kEKEYREVOKED, EKEYREVOKED);
    // 129
    TURBO_DECLARE_ERRNO(kEKEYREJECTED, EKEYREJECTED);
    // 130
    TURBO_DECLARE_ERRNO(kEOWNERDEAD, EOWNERDEAD);
    // 131
    TURBO_DECLARE_ERRNO(kENOTRECOVERABLE, ENOTRECOVERABLE);
    // 132
    TURBO_DECLARE_ERRNO(kERFKILL, ERFKILL);
    // 133
    TURBO_DECLARE_ERRNO(kEHWPOISON, EHWPOISON);
    // 134
    static constexpr StatusCode ESTOP = 134;
    TURBO_DECLARE_ERRNO(kESTOP, ESTOP);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    static constexpr StatusCode kMaxSystemErrno = ESTOP;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// placeholder for system errno
    // 135
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER2, 135);
    // 136
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER3, 136);
    // 137
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER4, 137);
    // 138
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER5, 138);
    // 139
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER6, 139);
    // 140
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER7, 140);
    // 141
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER8, 141);
    // 142
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER9, 142);
    // 143
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER10, 143);
    // 144
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER11, 144);
    // 145
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER12, 145);
    // 146
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER13, 146);
    // 147
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER14, 147);
    // 148
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER15, 148);
    // 149
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER16, 149);
    // 150
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER17, 150);
    // 151
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER18, 151);
    // 152
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER19, 152);
    // 153
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER20, 153);
    // 154
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER21, 154);
    // 155
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER22, 155);
    // 156
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER23, 156);
    // 157
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER24, 157);
    // 158
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER25, 158);
    // 159
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER26, 159);
    // 160
    TURBO_DECLARE_ERRNO(kEPLACEHOLDER27, 160);
    ////////////////////////////////////////////////////////////////////////////
    /// turbo error code
    static constexpr int kMinMapStatus = 161;
    // 161
    // StatusCode::kCancelled
    //
    // kCancelled  indicates the operation was cancelled,
    // typically by the caller.
    TURBO_DECLARE_ERRNO(kCancelled, 161);

    // 162
    // StatusCode::kUnknown
    //
    // kUnknown (gRPC code "UNKNOWN") indicates an unknown error occurred. In
    // general, more specific errors should be raised, if possible. Errors raised
    // by APIs that do not return enough error information may be converted to
    // this error.
    TURBO_DECLARE_ERRNO(kUnknown, 162);

    // 163
    // StatusCode::kInvalidArgument
    //
    // kInvalidArgument indicates the caller
    // specified an invalid argument, such as a malformed filename. Note that use
    // of such errors should be narrowly limited to indicate the invalid nature of
    // the arguments themselves. Errors with validly formed arguments that may
    // cause errors with the state of the receiving system should be denoted with
    // `kFailedPrecondition` instead.
    TURBO_DECLARE_ERRNO(kInvalidArgument, 163);

    // 164
    // StatusCode::kDeadlineExceeded
    //
    // kDeadlineExceeded indicates a deadline
    // expired before the operation could complete. For operations that may change
    // state within a system, this error may be returned even if the operation has
    // completed successfully. For example, a successful response from a server
    // could have been delayed long enough for the deadline to expire.
    TURBO_DECLARE_ERRNO(kDeadlineExceeded, 164);

    // 165
    // StatusCode::kNotFound
    //
    // kNotFound indicates some requested entity (such as
    // a file or directory) was not found.
    //
    // `kNotFound` is useful if a request should be denied for an entire class of
    // users, such as during a gradual feature rollout or undocumented allow list.
    // If a request should be denied for specific sets of users, such as through
    // user-based access control, use `kPermissionDenied` instead.
    TURBO_DECLARE_ERRNO(kNotFound, 165);

    // 166
    // StatusCode::kAlreadyExists
    //
    // kAlreadyExists indicates that the entity a
    // caller attempted to create (such as a file or directory) is already
    // present.
    TURBO_DECLARE_ERRNO(kAlreadyExists, 166);

    // 167
    // StatusCode::kPermissionDenied
    //
    // kPermissionDenied indicates that the caller
    // does not have permission to execute the specified operation. Note that this
    // error is different than an error due to an *un*authenticated user. This
    // error code does not imply the request is valid or the requested entity
    // exists or satisfies any other pre-conditions.
    //
    // `kPermissionDenied` must not be used for rejections caused by exhausting
    // some resource. Instead, use `kResourceExhausted` for those errors.
    // `kPermissionDenied` must not be used if the caller cannot be identified.
    // Instead, use `kUnauthenticated` for those errors.
    TURBO_DECLARE_ERRNO(kPermissionDenied, 167);

    // 168
    // StatusCode::kResourceExhausted
    //
    // kResourceExhausted indicates some resource
    // has been exhausted, perhaps a per-user quota, or perhaps the entire file
    // system is out of space.
    TURBO_DECLARE_ERRNO(kResourceExhausted, 168);

    // 169
    // StatusCode::kFailedPrecondition
    //
    // kFailedPrecondition indicates that the
    // operation was rejected because the system is not in a state required for
    // the operation's execution. For example, a directory to be deleted may be
    // non-empty, an "rmdir" operation is applied to a non-directory, etc.
    //
    // Some guidelines that may help a service implementer in deciding between
    // `kFailedPrecondition`, `kAborted`, and `kUnavailable`:
    //
    //  (a) Use `kUnavailable` if the client can retry just the failing call.
    //  (b) Use `kAborted` if the client should retry at a higher transaction
    //      level (such as when a client-specified test-and-set fails, indicating
    //      the client should restart a read-modify-write sequence).
    //  (c) Use `kFailedPrecondition` if the client should not retry until
    //      the system state has been explicitly fixed. For example, if a "rmdir"
    //      fails because the directory is non-empty, `kFailedPrecondition`
    //      should be returned since the client should not retry unless
    //      the files are deleted from the directory.
    TURBO_DECLARE_ERRNO(kFailedPrecondition, 169);

    // 170
    // StatusCode::kAborted
    //
    // kAborted indicates the operation was aborted,
    // typically due to a concurrency issue such as a sequencer check failure or a
    // failed transaction.
    //
    // See the guidelines above for deciding between `kFailedPrecondition`,
    // `kAborted`, and `kUnavailable`.
    TURBO_DECLARE_ERRNO(kAborted, 170);

    // 171
    // StatusCode::kOutOfRange
    //
    // kOutOfRange indicates the operation was
    // attempted past the valid range, such as seeking or reading past an
    // end-of-file.
    //
    // Unlike `kInvalidArgument`, this error indicates a problem that may
    // be fixed if the system state changes. For example, a 32-bit file
    // system will generate `kInvalidArgument` if asked to read at an
    // offset that is not in the range [0,2^32-1], but it will generate
    // `kOutOfRange` if asked to read from an offset past the current
    // file size.
    //
    // There is a fair bit of overlap between `kFailedPrecondition` and
    // `kOutOfRange`.  We recommend using `kOutOfRange` (the more specific
    // error) when it applies so that callers who are iterating through
    // a space can easily look for an `kOutOfRange` error to detect when
    // they are done.
    TURBO_DECLARE_ERRNO(kOutOfRange, 171);

    // 172
    // StatusCode::kUnimplemented
    //
    // kUnimplemented indicates the operation is not
    // implemented or supported in this service. In this case, the operation
    // should not be re-attempted.
    TURBO_DECLARE_ERRNO(kUnimplemented, 172);

    // 173
    // StatusCode::kInternal
    //
    // kInternal indicates an internal error has occurred
    // and some invariants expected by the underlying system have not been
    // satisfied. This error code is reserved for serious errors.
    TURBO_DECLARE_ERRNO(kInternal, 173);

    // 174
    // StatusCode::kUnavailable
    //
    // kUnavailable indicates the service is currently
    // unavailable and that this is most likely a transient condition. An error
    // such as this can be corrected by retrying with a backoff scheme. Note that
    // it is not always safe to retry non-idempotent operations.
    //
    // See the guidelines above for deciding between `kFailedPrecondition`,
    // `kAborted`, and `kUnavailable`.
    TURBO_DECLARE_ERRNO(kUnavailable, 174);

    // 175
    // StatusCode::kDataLoss
    //
    // kDataLoss indicates that unrecoverable data loss or
    // corruption has occurred. As this error is serious, proper alerting should
    // be attached to errors such as this.
    TURBO_DECLARE_ERRNO(kDataLoss, 175);

    // 176
    // StatusCode::kUnauthenticated
    //
    // kUnauthenticated indicates that the request
    // does not have valid authentication credentials for the operation. Correct
    // the authentication and try again.
    TURBO_DECLARE_ERRNO(kUnauthenticated, 176);

    // 177
    TURBO_DECLARE_ERRNO(kTryAgain, 177);

    // 178
    TURBO_DECLARE_ERRNO(kAlreadyStop, 178);

    // 179
    TURBO_DECLARE_ERRNO(kResourceBusy, 179);

    static constexpr int kMaxMapStatus = kResourceBusy;

    /**
     * @ingroup turbo_base_status
     * @brief Returns the StatusCode for `error_number`, which should be an `errno` value.
     *        See https://en.cppreference.com/w/cpp/error/errno_macros and similar
     *        references.
     * @param error_number The error number.
     * @return StatusCode
     */
    turbo::StatusCode errno_to_status_code(int error_number);

}  // namespace turbo

#endif // TURBO_STATUS_STATUS_CODE_H_
