# PyOpenCL Inline Comments Tutorial

This is an introduction to parallel computation with Python and OpenCL.  The tutorial is composed of a numbered series of files.  Each file contains a short, complete script with extensive inline comments.

## About The Comments

This tutorial is an expiriment in code education.  The idea is that commenting each line will force the author to slow down and use plain english to explain every detail in a way that a diligent reader can understand.

I have tried to write the code and the comments in a way that they could be separated and each one would still tell the complete story of what is going on.  With two sources of information, hopefully readers will seldom have to break focus and look something up.

This tutorial does not follow PEP 8's recommendation of 79 characters per-line because there are long comments mixed in with the code.  It is best to turn on code-wrap in your editor while you are reading.

## About The Tutorial

I am writing this tutorial to learn and have some fun.  A lot of my day-to-day work is web programming.  That is enjoyable, but there is a limited amount of craftsmanship and original thought required.  PyOpenCL is a different story:  Python brings clarity to the table, while OpenCL provides access to all the power modern hardware can deliver.  Together these two languages are a double-bladed lightsaber in a world of butter knives.

# Index

- 01 Introspection - Find out about your computer's OpenCL situation
- 02 Array Sum - Use OpenCL To Add Two Large Random Arrays
- 03 Array Sum - Use OpenCL To Add Two Large Random Array (Using PyOpenCL Arrays)