static PyObject* pybv_bv_preprocess1(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace bv;

    {
    PyObject* pyobj_mat = NULL;
    Mat mat;
    Mat retval;

    const char* keywords[] = { "mat", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:preprocess1", (char**)keywords, &pyobj_mat) &&
        pybv_to(pyobj_mat, mat, ArgInfo("mat", 0)) )
    {
        ERRWRAP2(retval = bv::preprocess1(mat));
        return pybv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_mat = NULL;
    Mat mat;
    Mat retval;

    const char* keywords[] = { "mat", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:preprocess1", (char**)keywords, &pyobj_mat) &&
        pybv_to(pyobj_mat, mat, ArgInfo("mat", 0)) )
    {
        ERRWRAP2(retval = bv::preprocess1(mat));
        return pybv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pybv_bv_preprocess2(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace bv;

    {
    PyObject* pyobj_mat = NULL;
    Mat mat;
    Mat retval;

    const char* keywords[] = { "mat", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:preprocess2", (char**)keywords, &pyobj_mat) &&
        pybv_to(pyobj_mat, mat, ArgInfo("mat", 0)) )
    {
        ERRWRAP2(retval = bv::preprocess2(mat));
        return pybv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_mat = NULL;
    Mat mat;
    Mat retval;

    const char* keywords[] = { "mat", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:preprocess2", (char**)keywords, &pyobj_mat) &&
        pybv_to(pyobj_mat, mat, ArgInfo("mat", 0)) )
    {
        ERRWRAP2(retval = bv::preprocess2(mat));
        return pybv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pybv_bv_preprocess3(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace bv;

    {
    PyObject* pyobj_mat = NULL;
    Mat mat;
    vector <vector<Rect> > retval;

    const char* keywords[] = { "mat", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:preprocess3", (char**)keywords, &pyobj_mat) &&
        pybv_to(pyobj_mat, mat, ArgInfo("mat", 0)) )
    {
        ERRWRAP2(retval = bv::preprocess3(mat));
        return pybv_from(retval);
    }
    }
    PyErr_Clear();

    {
    PyObject* pyobj_mat = NULL;
    Mat mat;
    vector <vector<Rect> > retval;

    const char* keywords[] = { "mat", NULL };
    if( PyArg_ParseTupleAndKeywords(args, kw, "O:preprocess3", (char**)keywords, &pyobj_mat) &&
        pybv_to(pyobj_mat, mat, ArgInfo("mat", 0)) )
    {
        ERRWRAP2(retval = bv::preprocess3(mat));
        return pybv_from(retval);
    }
    }

    return NULL;
}

static PyObject* pybv_bv_preprocess4(PyObject* , PyObject* args, PyObject* kw)
{
    using namespace bv;

    vector<vector<vector<int> > > retval;

    if(PyObject_Size(args) == 0 && (kw == NULL || PyObject_Size(kw) == 0))
    {
        ERRWRAP2(retval = bv::preprocess4());
        return pybv_from(retval);
    }

    return NULL;
}

