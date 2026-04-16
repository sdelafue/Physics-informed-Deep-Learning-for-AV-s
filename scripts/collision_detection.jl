"""
    detect_collision(ego_trajectory::Matrix{Float64},
                     object_trajectory::Matrix{Float64};
                     collision_radius::Float64=2.0) → Bool

Checks whether the predicted trajectory of a detected object crosses into
the ego vehicle's forward collision zone at any of the 6 future timesteps.

# Arguments
- `ego_trajectory`:    6×7 matrix — ego vehicle trajectory (6 timesteps)
- `object_trajectory`: 6×7 matrix — predicted object trajectory (6 timesteps)
- `collision_radius`:  forward detection radius in meters (default: 2.0)

# Matrix column layout
    Col 1: dt [s]  |  Col 2: x [m]  |  Col 3: y [m]  |  Col 4: yaw [rad]
    Col 5: vx [m/s]  |  Col 6: vy [m/s]  |  Col 7: yaw_rate [rad/s]

# Collision zone
A 90° circular sector (±45° from the ego's heading) extending
`collision_radius` meters in front of the vehicle. Objects behind
the vehicle are ignored since they cannot cause a forward collision.

# Returns
`true` if a collision is predicted at any timestep, `false` otherwise.
"""
function detect_collision(ego_trajectory::Matrix{Float64},
                          object_trajectory::Matrix{Float64};
                          collision_radius::Float64=2.0)::Bool
    n_steps = size(ego_trajectory, 1)

    for t in 1:n_steps
        ego_x   = ego_trajectory[t, 2]
        ego_y   = ego_trajectory[t, 3]
        ego_yaw = ego_trajectory[t, 4]

        obj_x = object_trajectory[t, 2]
        obj_y = object_trajectory[t, 3]

        dx = obj_x - ego_x
        dy = obj_y - ego_y

        dist = sqrt(dx^2 + dy^2)
        if dist > collision_radius
            continue
        end

        # Bearing from ego to object
        bearing = atan(dy, dx)

        # Angular difference normalized to [-π, π]
        angle_diff = atan(sin(bearing - ego_yaw), cos(bearing - ego_yaw))

        # Within ±45° forward cone
        if abs(angle_diff) <= π / 4
            return true
        end
    end

    return false
end
