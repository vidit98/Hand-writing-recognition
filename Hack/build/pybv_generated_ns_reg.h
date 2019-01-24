static PyMethodDef methods_bv[] = {
    {"preprocess1", (PyCFunction)pybv_bv_preprocess1, METH_VARARGS | METH_KEYWORDS, "preprocess1(mat) -> retval\n."},
    {"preprocess2", (PyCFunction)pybv_bv_preprocess2, METH_VARARGS | METH_KEYWORDS, "preprocess2(mat) -> retval\n."},
    {"preprocess3", (PyCFunction)pybv_bv_preprocess3, METH_VARARGS | METH_KEYWORDS, "preprocess3(mat) -> retval\n."},
    {"preprocess4", (PyCFunction)pybv_bv_preprocess4, METH_VARARGS | METH_KEYWORDS, "preprocess4() -> retval\n."},
    {NULL, NULL}
};

static ConstDef consts_bv[] = {
    {NULL, 0}
};

static void init_submodules(PyObject * root) 
{
  init_submodule(root, MODULESTR"", methods_bv, consts_bv);
};
