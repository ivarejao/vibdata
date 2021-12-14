def test_download():
    from vibdata.datahandler import MFPT_raw
    MFPT_raw("/tmp/", download=True)


def test_metainfo():
    from vibdata.datahandler import MFPT_raw
    D = MFPT_raw("/tmp/", download=True)
    assert(len(D.getMetaInfo()) == 20)
