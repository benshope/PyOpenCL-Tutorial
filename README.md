# PyOpenCL Inline Comments Tutorial

This is an introduction to parallel computation with Python and OpenCL.  The tutorial is composed of a numbered series of files.  Each file contains a short, complete script with extensive inline comments.

## About The Comments

This tutorial is an expiriment in code education.  The idea is that commenting each line will force the author to slow down and use plain english to explain every detail in a way that a diligent reader can understand.

I have tried to write the code and the comments in a way that they could be separated and each one would still tell the complete story of what is going on.  With two sources of information, hopefully readers will seldom have to break focus and look something up.

This tutorial does not follow PEP 8's recommendation of 79 characters per-line because there are long comments mixed in with the code.  It is best to turn on code-wrap in your editor while you are reading.

## About The Tutorial

I am writing this tutorial because PyOpenCL is a combination of tools that is worth learning.  Python allows exceptional clarity-of-expression while OpenCL provides access to all the power modern hardware can deliver.  Together these two languages are like a lightsaber in a world of butter knives.

# Index

- 01 Introspection - Find out about your computer's OpenCL situation
- 02 Array Sum - Use OpenCL To Add Two Large Random Arrays
- 03 Array Sum - Use OpenCL To Add Two Large Random Array (Using PyOpenCL Arrays)