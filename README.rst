SVG Logo Animation using Machine Learning
-----------------------------------------

This project allows to automatically animate logos in SVG format using machine learning.

Its functionality includes:

- Scrape SVG logos from Wikidata
- Extract SVG information, e.g., size, position, color
- Get SVG embeddings of logos by using deepSVG's hierarchical generative network
- Automatically animate logos using two different approaches: Genetic algorithm and entmoot optimizer


The project started in November 2020 as a Masters Team Project at the University of Mannheim.


Data Aquisition
^^^^^^^^^^^^^^^

The data is collected through a `labeling website <https://animate-logos.web.app/>`__ (`Github <https://github.com/J4K08L4N63N84HN/animate_logos_label_website>`__) where users can rate the quality of animations.


Documentation and Usage
^^^^^^^^^^^^^^^^^^^^^^^

Animate your logos with machine learning by using the following code or visit our `website <https://animate-logos.herokuapp.com/>`__ (`Github <https://github.com/J4K08L4N63N84HN/animate_logos_website>`__), where you can get an animated version of your uploaded logo.

.. code:: python

    from src.pipeline import Logo
    logo = Logo(data_dir='path/to/my/svgs/logo.svg')
    logo.animate()

Detailed documentation and usage instructions can be found `here <https://animate-logos.readthedocs.io/en/latest/>`__.


