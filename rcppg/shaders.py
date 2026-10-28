import pyqtgraph.opengl as gl

gl.shaders.Shaders.append(gl.shaders.ShaderProgram('balloon_glow', 
    [   
        gl.shaders.VertexShader("""
                uniform mat4 u_mvp;
                uniform mat3 u_normal;
                attribute vec4 a_position;
                attribute vec3 a_normal;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    v_normal = normalize(u_normal * a_normal);
                    v_color = a_color;
                    gl_Position = u_mvp * a_position;
                }
            """),
    gl.shaders.FragmentShader("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    vec4 color = v_color;
                    color.w = min(color.w + 4.0 * color.w * pow(v_normal.x*v_normal.x + v_normal.y*v_normal.y, 20.0), 1.0);
                    gl_FragColor = color;
                }
            """)
    ]))
