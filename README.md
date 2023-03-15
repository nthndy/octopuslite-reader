# DaskOctopus

```
Load multidimensional image stacks using lazy loading.

A simple class to load OctopusLite data from a directory. Caches data once
it is loaded to prevent excessive I/O to the data server. Can directly
address different channels using the `Channels` enumerator.

This branch (octopusheavy) has additional capacity for larger multidimensional
image volumes as well as tiling mosaics of images together using elements from
DaskFusion.

Usage
-----
>>> from octopuslite import DaskOctopus, MetadataParser
>>> images =  DaskOctopus(
    path = '/path/to/your/data/',
    crop = (1200,1600),
    transforms = 'path/to/transform_array.npy',
    remove_background = True,
    parser = MetadataParser.OCTOPUS,
)
>>> gfp = images["GFP"]
```
