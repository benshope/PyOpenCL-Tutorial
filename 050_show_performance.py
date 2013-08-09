import pyopencl as cl
import pyopencl.characterize.performance as performance

context = cl.create_some_context()
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

overhead, latency = performance.get_profiling_overhead(context)

# XXX Find out the difference between {} and {0} XXX
# XXX Turn off scientific notation for second measurements?  XXX

# XXX Find out what the hell all these numbers mean XXX
print("\n\nCommand Latency: {} s".format(latency))
print("Profiling Overhead: {} s -> {}".format(overhead, 100*overhead/latency))  

# XXX Both these lines appear to break the program on a Mac, fix them XXX
# print("Empty Kernel: {} s".format(performance.get_empty_kernel_time(queue)))
# print("Float32 Add: {} GOps/s\n".format(performance.get_add_rate(queue)/1e9))

for transfer_type in [
        performance.HostToDeviceTransfer,
        performance.DeviceToHostTransfer,
        performance.DeviceToDeviceTransfer]:
    
    print("\n" + transfer_type.__name__)
    print("    Latency: {0} s".format(performance.transfer_latency(queue, transfer_type)))
    for exponent in range(6, 28, 2):
        bytes = 1<<exponent  # Two to the power of exponent XXX Is exponent the right word? XXX
        print("    Bandwidth at {0} Bytes: {1} GB/s".format(bytes, performance.transfer_bandwidth(queue, transfer_type, bytes)/1e9))
