from libc.stdio cimport *
import numpy as np
cimport numpy as np

cdef extern from "stdio.h":
    FILE *fopen   (const_char *FILENAME, const_char  *OPENTYPE)
    int fscanf   (FILE *STREAM, const_char *TEMPLATE, ...)
    char *fgets (char *S, int COUNT, FILE *STREAM)
    int fgetc   (FILE *stream)
    int      fseek  (FILE *STREAM, long int OFFSET, int WHENCE)
    int printf   (const_char *TEMPLATE, ...)
    int  puts   (const_char *S)
    enum: EOF
    enum: SEEK_SET
    enum: SEEK_CUR
    enum: SEEK_END
    # int sscanf   (const_char *S, const_char *TEMPLATE, ...)
    # int printf   (const_char *TEMPLATE, ...)

DEF MAX_ROW = 108676

# def c_parse_records_fs(const_char *f_name):
#     cdef int node_id
#     cdef char[10] prot
#     cdef char[10] node
#     cdef int duration
#     cdef FILE* cfile
#     cdef double start_time, end_time
#     cdef double t3
#     cdef int s1, s2, s3, s4
#     cdef int d1, d2, d3, d4
#     cdef int port1, port2
#     cdef int flow_size

#     cdef np.ndarray flows = np.ndarray((MAX_ROW,), dtype= np.dtype([
#         ('start_time', np.float64, 1),
#         ('end_time', np.float64, 1),
#         ('src_ip', np.uint8, (4,)),
#         ('src_port', np.int16, 1),
#         ('dst_ip', np.uint8, (4,)),
#         ('dst_port', np.int16, 1),
#         ('prot', np.str_, 5),
#         ('node', np.str_ , 5),
#         ('flow_size', np.float64, 1),
#         ('duration', np.float64, 1),
#         ])
#     )

#     cfile = fopen(f_name, 'r') # attach the stream
#     i = -1
#     while True:
#         i += 1
#         if i > MAX_ROW:
#             raise Exception("MAX_ROW too SMALL! Please increase MAX_ROW in "
#                             "CythonUtil.pyx")

#         value = fscanf(cfile, 
#                'textexport n%i '
#                '%lf %lf %lf %i.%i.%i.%i:%i->%i.%i.%i.%i:%i %s 0x0 '
#                '%s %i %i FSA\n',
#                 &node_id, 
#                 &start_time, &end_time, &t3, 
#                 &s1, &s2, &s3, &s4, &port1,
#                 &d1, &d2, &d3, &d4, &port2,
#                 &prot[0], &node[0], 
#                 &duration,
#                 &flow_size
#                 )

#         if value == EOF:
#             break
#         elif value != 18:
#             raise Exception("value = " + str(value))

#         flows[i] = (start_time, end_time, (s1, s2, s3, s4), port1, 
#                 (d1, d2, d3, d4), port2, prot, node, flow_size, duration)

#     return flows, i


def c_parse_records_tshark(char *f_name):
    cdef FILE* cfile
    cdef double time
    cdef char[10] prot
    cdef char[16] src_ip
    cdef char[16] dst_ip
    cdef double sz
    cdef char[10] seq
    cdef int value = 0
    cdef long src_port, dst_port
    cdef char tmp = 'c'

    cdef np.ndarray flows = np.ndarray((MAX_ROW,), dtype= np.dtype([
        ('start_time', np.float64),
        ('src_ip', np.str_, 15),
        ('dst_ip', np.str_, 15),
        ('prot', np.str_, 5),
        ('size', np.float64),
        ])
    )

    cfile = fopen(f_name, 'r') # attach the stream
    i = -1
    while True:
        i += 1
        if i > MAX_ROW:
            raise Exception("MAX_ROW too SMALL! Please increase MAX_ROW in "
                            "CythonUtil.pyx")

        value = fscanf(cfile, '%s %lf %s -> %s',
                seq, &time, src_ip, dst_ip)
        if value == EOF:
            break

        fseek(cfile, 2, SEEK_CUR)
        tmp = fgetc(cfile)
        if tmp == ' ':
            value += fscanf(cfile, '%lf %s\n', &sz, prot)
        else:
            fseek(cfile, -2, SEEK_CUR)
            value += fscanf(cfile, '%ld %ld %lf %s\n', &src_port, &dst_port, &sz, prot)
            # printf('%ld\n%ld\n%lf\n%s\n', src_port, dst_port, sz, prot)
            # print('time', time)
            # printf('%s\n%s\n%s\n%s\n', prot2, prot3, prot4, prot)


        if value != 6 and value != 8:
            raise Exception("value = " + str(value))
        else:
            value = 0

        flows[i] = (time, src_ip, dst_ip, prot, sz)

        # printf('textexport n%i '
        #        '%lf %lf %lf %i.%i.%i.%i:%i->%i.%i.%i.%i:%i tcp 0x0 '
        #        '%s %i %i FSA\n',
        #         node_id, 
        #         start_time2, end_time2, t3, 
        #         s1, s2, s3, s4, port1,
        #         d1, d2, d3, d4, port2,
        #         prot, tp_unknown,
        #         flow_size
        #         ) 

        # break
    # print 'hello world'
    return flows, i
