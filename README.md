# lethal-heat

<a name="readme-top"></a>


<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT HEADER -->
<br />
<div align="center">
  <a href="https://github.com/robert-edwin-rouse/reclassifying-lethal-heat">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Streamflow Prediction Using Artificial Neural Networks &amp; Soil Moisture Proxies
</h3>

  <p align="center">
    <br />
    <a href="https://github.com/robert-edwin-rouse/reclassifying-lethal-heat/issues">Report Bug</a>
    ·
    <a href="https://github.com/robert-edwin-rouse/reclassifying-lethal-heat/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](./figures/.png)

This codebase accompanies the paper 'Reclassifying Lethal Heat'.  It includes the code to download the requisite ERA5 data from the ECMWF Copernicus Data Store, the code to run the model and reproduce all of the results, and reproduce the main figures from the paper.

In order to maximise flexibility, core functions are shared between this project and related projects through the apollo environmental data science submodule; the version required to reproduce the results in the paper is included.

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/robert-edwin-rouse/reclassifying-lethal-heat.svg?style=for-the-badge
[contributors-url]: https://github.com/robert-edwin-rouse/reclassifying-lethal-heat/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/robert-edwin-rouse/reclassifying-lethal-heat.svg?style=for-the-badge
[forks-url]: https://github.com/robert-edwin-rouse/reclassifying-lethal-heat/network/members
[stars-shield]: https://img.shields.io/github/stars/robert-edwin-rouse/reclassifying-lethal-heat.svg?style=for-the-badge
[stars-url]: https://github.com/robert-edwin-rouse/reclassifying-lethal-heat/stargazers
[issues-shield]: https://img.shields.io/github/issues/robert-edwin-rouse/reclassifying-lethal-heat.svg?style=for-the-badge
[issues-url]: https://github.com/robert-edwin-rouse/reclassifying-lethal-heat/issues
[license-shield]: https://img.shields.io/github/license/robert-edwin-rouse/reclassifying-lethal-heat.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[product-screenshot]: ./figures/.png