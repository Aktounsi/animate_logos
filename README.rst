SVG Logo Animation using Machine Learning
-----------------------------------------

This project allows to automatically animate logos in SVG format using Machine Learning.

Its functionality includes:

- Scrape SVG logos from Wikidata
- Extract SVG information, e.g., size, position, color
- Get SVG embeddings of logos by using deepSVG's hierarchical generative network
- Automatically animate logos using two different approaches: Genetic algorithm and entmoot optimizer


The project started in November 2020 as a Masters Team Project at the University of Mannheim.


Documentation and Usage
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from src.pipeline import Logo
    logo = Logo(data_dir='path/to/my/svgs/logo.svg')
    logo.animate()


Detailed documentation and usage instructions can be found
`here <https://animate-logos.readthedocs.io/en/latest/>`__.
