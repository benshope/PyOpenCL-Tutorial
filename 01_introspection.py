# Learn out about your system

import pyopencl as cl

print('\n\nOpenCL Platforms and Devices:')
for platform in cl.get_platforms():
    print('=' * 60)
    print('Platform Name:  ' + platform.name)
    print('Platform Profile:  ' + platform.profile)
    print('Platform Vendor:  ' + platform.vendor)
    print('Platform Version:  ' + platform.version)
    for device in platform.get_devices():
        print('    ' + '-' * 56)
        print('    Device Name:  ' + device.name)
        print('    Device Type:  ' + cl.device_type.to_string(device.type))
        print('    Device Memory: {0} GB'.format(device.global_mem_size/1073741824.0))
        print('    Device Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
        print('    Device Compute Units:  {0}'.format(device.max_compute_units))
print('\n')