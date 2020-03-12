import numpy as np

def generate_point_clouds(n_samples, n_points, eps):
    sphere_point_clouds = [
        np.asarray(
            [
                [
                    np.cos(s) * np.cos(t) + eps * (np.random.rand(1)[0] - 0.5),
                    np.cos(s) * np.sin(t) + eps * (np.random.rand(1)[0] - 0.5),
                    np.sin(s) + eps * (np.random.rand(1)[0] - 0.5),
                ]
                for t in range(n_points)
                for s in range(n_points)
            ]
        )
        for kk in range(n_samples)
    ]
    # label spheres with 0
    sphere_labels = np.zeros(n_samples)

    torus_point_clouds = [
        np.asarray(
            [
                [
                    (2 + np.cos(s)) * np.cos(t) + eps * (np.random.rand(1)[0] - 0.5),
                    (2 + np.cos(s)) * np.sin(t) + eps * (np.random.rand(1)[0] - 0.5),
                    np.sin(s) + eps * (np.random.rand(1)[0] - 0.5),
                ]
                for t in range(n_points)
                for s in range(n_points)
            ]
        )
        for kk in range(n_samples)
    ]
    # label tori with 1
    torus_labels = np.ones(n_samples)

    point_clouds = np.concatenate((sphere_point_clouds, torus_point_clouds))
    labels = np.concatenate((sphere_labels, torus_labels))

    return point_clouds, labels
