"""

"""
template = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   version="1.1"
   id="svg2"
   viewBox="0 0 337.5884 434.87447"
   height="122.73124mm"
   width="95.274948mm">
  <defs
     id="defs4" />
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     transform="translate(-159.34241,-265.3449)"
     id="layer1">
    <text
       id="text10"
       y="700.21936"
       x="148.56522"
       style="font-style:normal;font-weight:normal;font-size:596.53240967px;line-height:125%;font-family:sans-serif;letter-spacing:0px;word-spacing:0px;fill:#000000;fill-opacity:1;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       xml:space="preserve"><tspan
         style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-family:'DejaVu Sans Mono';-inkscape-font-specification:'DejaVu Sans Mono'"
         y="700.21936"
         x="148.56522"
         id="tspan12">{}</tspan></text>
  </g>
</svg>
"""

svg = template.format("A")
with open("/tmp/output.svg", "w") as fh:
    fh.write(svg)