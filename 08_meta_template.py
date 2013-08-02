import pyopencl as cl
import numpy
from mako.template import Template

local_size = 256
thread_strides = 32
macroblock_count = 33
dtype = numpy.float32
total_size = local_size*thread_strides*macroblock_count

context = cl.create_some_context()
queue = cl.CommandQueue(context)

a = numpy.random.randn(total_size).astype(dtype)
b = numpy.random.randn(total_size).astype(dtype)
c = numpy.empty_like(a)

a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b.nbytes)

template = Template("""
    __kernel void add(
            __global ${ type_name } *tgt, 
            __global const ${ type_name } *op1, 
            __global const ${ type_name } *op2)
    {
      int idx = get_local_id(0)
        + ${ local_size } * ${ thread_strides }
        * get_group_id(0);

      % for i in range(thread_strides):
          <% offset = i*local_size %>
          tgt[idx + ${ offset }] = 
            op1[idx + ${ offset }] 
            + op2[idx + ${ offset } ];
      % endfor
    }""")

rendered_template = template.render(type_name="float", local_size=local_size, thread_strides=thread_strides)

kernel = cl.Program(context, str(rendered_template)).build().add

kernel(queue, (local_size*macroblock_count,), (local_size,), c_buf, a_buf, b_buf)

cl.enqueue_read_buffer(queue, c_buf, c).wait()

print("a: {0}".format(a))
print("b: {0}".format(b))
print("c: {0}".format(c))  # Print all three arrays, to show sum() worked