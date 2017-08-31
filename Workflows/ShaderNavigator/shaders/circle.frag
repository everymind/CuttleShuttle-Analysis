#version 400
in vec2 texCoord;
out vec4 fragColor;
uniform vec4 color;
uniform float radius;
uniform vec2 center;

void main()
{
  float dist = length(texCoord - center * 0.5 - 0.5);
  fragColor = dist < radius ? color : 0.0;
}
