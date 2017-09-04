#version 400
in vec2 texCoord;
out vec4 fragColor;
uniform vec4 color;

void main()
{
  float dist = length(texCoord*2-1);
  fragColor = dist < 1 ? color : 0.0;
}
