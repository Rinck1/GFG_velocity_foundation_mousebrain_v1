all_edges = {
    "DentateGyrus.h5ad": [("nIPC", "Neuroblast"), ("Neuroblast", "Granule immature"), ("Granule immature", "Granule mature"),
                 ('OPC', 'OL')], 
    "MouseBrain.h5ad": [
        ('RG, Astro, OPC', 'IPC'), ('IPC', 'Subplate'), ('IPC', 'Deeper Layer'), ('IPC', 'Upper Layer'), ('RG, Astro, OPC', 'Ependymal cells')
        # ('RG, Astro, OPC', 'IPC'), ('IPC', 'V-SVZ'), ('V-SVZ', 'Upper layer'), ('Upper layer', 'Deeper layer')
    ],
    "Pancreas.h5ad": [("Pre-endocrine", "Alpha"), ("Pre-endocrine", "Beta"), ("Pre-endocrine", "Delta"), 
                 ("Pre-endocrine", "Epsilon")],
    "endocrinogenesis_day15.h5ad": [
        ("Ductal", "Pre-endocrine"),
        ("Pre-endocrine", "Ngn3 low EP"),
        ("Ngn3 low EP", "Ngn3 high EP"),
        ("Ngn3 high EP", "Alpha"),
        ("Ngn3 high EP", "Beta"),
        ("Ngn3 high EP", "Delta"),
        ("Ngn3 high EP", "Epsilon")
    ],
    "Hindbrain_GABA_Glio.h5ad": [('Neural stem cells', 'Proliferating VZ progenitors'), ('Proliferating VZ progenitors', 'VZ progenitors'), 
                 ('VZ progenitors', 'Differentiating GABA interneurons'), ('VZ progenitors', 'Gliogenic progenitors'), 
                 ('Differentiating GABA interneurons', 'GABA interneurons')],
    "erythroid_lineage.h5ad": [
        ('Blood progenitors 1', 'Blood progenitors 2'),
        ('Blood progenitors 2', 'Erythroid1'),
        ('Erythroid1', 'Erythroid2'),
        ('Erythroid2', 'Erythroid3')
    ], 
    "organoids.h5ad":  [("Stem cells", "TA cells"), ("Stem cells", "Goblet cells")],
    "retina.h5ad": [('Progenitor', 'Neuroblast'), ('Neuroblast', 'PR'), ('Neuroblast', 'AC/HC'), ('Neuroblast', 'RGC')], 
    "reprogramming.h5ad": [('5', '2'), ('5', '8'), ('2', '7'), ('8', '4'), ('7', '6'), ('4', '9'), ('6', '1'), 
                 ('9', '8'), ('6', '0'), ('8', '0')],
    "hematopoiesis.h5ad": [('HSC', 'GMP-like'), ('HSC', 'MEP-like'), ('GMP-like', 'Mon'), ('GMP-like', 'Neu'), ('GMP-like', 'Bas'), ('MEP-like', 'Ery'), ('MEP-like', 'Meg')]
}