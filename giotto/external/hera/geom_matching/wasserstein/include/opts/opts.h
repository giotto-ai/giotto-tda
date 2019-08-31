/**
 * Author: Dmitriy Morozov <dmitriy@mrzv.org>
 * The interface is heavily influenced by GetOptPP (https://code.google.com/p/getoptpp/).
 */

#ifndef OPTS_OPTS_H
#define OPTS_OPTS_H

#include <iostream>
#include <sstream>
#include <string>
#include <list>
#include <vector>

namespace opts {

// Converters
template<class T>
struct Converter
{
                    Converter()                     {}
    static
    T               convert(const std::string& val) { std::istringstream iss(val); T res; iss >> res; return res; }
};

// Type
template<class T>
struct Traits
{
    static std::string  type_string()               { return "UNKNOWN TYPE"; }
};

template<>
struct Traits<int>
{
    static std::string  type_string()               { return "INT"; }
};

template<>
struct Traits<short int>
{
    static std::string  type_string()               { return "SHORT INT"; }
};

template<>
struct Traits<unsigned>
{
    static std::string  type_string()               { return "UNSIGNED INT"; }
};

template<>
struct Traits<short unsigned>
{
    static std::string  type_string()               { return "SHORT UNSIGNED INT"; }
};

template<>
struct Traits<float>
{
    static std::string  type_string()               { return "FLOAT"; }
};

template<>
struct Traits<double>
{
    static std::string  type_string()               { return "DOUBLE"; }
};

template<>
struct Traits<std::string>
{
    static std::string  type_string()               { return "STRING"; }
};


struct BasicOption
{
                    BasicOption(char        s_,
                                std::string l_,
                                std::string default_,
                                std::string type_,
                                std::string help_):
                        s(s_), l(l_), d(default_), t(type_), help(help_)                    {}

    int             long_size() const                           { return l.size() + 1 + t.size(); }

    void            output(std::ostream& out, int max_long) const
    {
        out << "   ";
        if (s)
            out << '-' << s << ", ";
        else
            out << "    ";

        out << "--" << l << ' ';

        if (!t.empty())
            out << t;

        for (int i = long_size(); i < max_long; ++i)
            out <<  ' ';

        out << "   " << help;

        if (!d.empty())
        {
            out << " [default: " << d << "]";
        }
        out << '\n';
    }

    char            s;
    std::string     l;
    std::string     d;
    std::string     t;
    std::string     help;
};

// Option
template<class T>
struct OptionContainer: public BasicOption
{
                    OptionContainer(char               s_,
                                    const std::string& l_,
                                    T&                 var_,
                                    const std::string& help_,
                                    const std::string& type_ = Traits<T>::type_string()):
                        BasicOption(s_, l_, default_value(var_), type_, help_),
                        var(&var_)                  {}

    static
    std::string     default_value(const T& def)
    {
        std::ostringstream oss;
        oss << def;
        return oss.str();
    }

    void            parse(std::list<std::string>& args) const
    {
        std::string short_opt = "-"; short_opt += s;
        std::string long_opt  = "--" + l;
        for (std::list<std::string>::iterator cur = args.begin(); cur != args.end(); ++cur)
        {
            if (*cur == short_opt || *cur == long_opt)
            {
                cur = args.erase(cur);
                if (cur != args.end())
                {
                    *var = Converter<T>::convert(*cur);
                    cur = args.erase(cur);
                    break;              // finds first occurrence
                }
                else
                     break;         // if the last option's value is missing, it remains default

            }
        }
    }

    T*  var;
};

template<class T>
struct OptionContainer< std::vector<T> >: public BasicOption
{
                    OptionContainer(char               s_,
                                    const std::string& l_,
                                    std::vector<T>&    var_,
                                    const std::string& help_,
                                    const std::string& type_ = "SEQUENCE"):
                        BasicOption(s_, l_, default_value(var_), type_, help_),
                        var(&var_)                  { }

    static
    std::string     default_value(const std::vector<T>& def)
    {
        std::ostringstream oss;
        oss << "(";
        if (def.size())
            oss << def[0];
        for (int i = 1; i < def.size(); ++i)
            oss << ", " << def[i];
        oss << ")";
        return oss.str();
    }

    void            parse(std::list<std::string>& args) const
    {
        std::string short_opt = "-"; short_opt += s;
        std::string long_opt  = "--" + l;
        for (std::list<std::string>::iterator cur = args.begin(); cur != args.end(); ++cur)
        {
            if (*cur == short_opt || *cur == long_opt)
            {
                cur = args.erase(cur);
                if (cur != args.end())
                {
                    var->push_back(Converter<T>::convert(*cur));
                    cur = args.erase(cur);
                }
                --cur;
            }
        }
    }

    std::vector<T>* var;
};


template<class T>
OptionContainer<T>
Option(char s, const std::string& l, T& var, const std::string& help)       { return OptionContainer<T>(s, l, var, help); }

template<class T>
OptionContainer<T>
Option(char s, const std::string& l, T& var,
       const std::string& type, const std::string& help)                    { return OptionContainer<T>(s, l, var, help, type); }

template<class T>
OptionContainer<T>
Option(const std::string& l, T& var, const std::string& help)               { return OptionContainer<T>(0, l, var, help); }

template<class T>
OptionContainer<T>
Option(const std::string& l, T& var,
       const std::string& type, const std::string& help)                    { return OptionContainer<T>(0, l, var, help, type); }

// Present
struct PresentContainer: public BasicOption
{
                PresentContainer(char s, const std::string& l, const std::string& help):
                    BasicOption(s,l,"","",help)           {}
};

inline
PresentContainer
Present(char s, const std::string& l, const std::string& help)              { return PresentContainer(s, l, help); }

inline
PresentContainer
Present(const std::string& l, const std::string& help)                      { return PresentContainer(0, l, help); }

// PosOption
template<class T>
struct PosOptionContainer
{
                PosOptionContainer(T& var_):
                    var(&var_)                                              {}

    bool        parse(std::list<std::string>& args) const
    {
        if (args.empty())
            return false;

        *var = Converter<T>::convert(args.front());
        args.pop_front();
        return true;
    }

    T*          var;
};

template<class T>
PosOptionContainer<T>
PosOption(T& var)                                                           { return PosOptionContainer<T>(var); }


// Options
struct Options
{
            Options(int argc_, char** argv_):
                args(argv_ + 1, argv_ + argc_),
                failed(false)                       {}

    template<class T>
    Options&    operator>>(const OptionContainer<T>&  oc);
    bool        operator>>(const PresentContainer&    pc);
    template<class T>
    Options&    operator>>(const PosOptionContainer<T>& poc);

                operator bool()                     { return !failed; }


    friend
    std::ostream&
    operator<<(std::ostream& out, const Options& ops)
    {
        int max_long = 0;
        for (std::list<BasicOption>::const_iterator cur =  ops.options.begin();
                                                    cur != ops.options.end();
                                                  ++cur)
        {
            int cur_long = cur->long_size();
            if (cur_long > max_long)
                max_long = cur_long;
        }

        out << "Options:\n";
        for (std::list<BasicOption>::const_iterator cur =  ops.options.begin();
                                                    cur != ops.options.end();
                                                  ++cur)
            cur->output(out, max_long);

        return out;
    }


    private:
        std::list<std::string>                      args;
        std::list<BasicOption>                      options;
        bool                                        failed;
};

template<class T>
Options&
Options::operator>>(const OptionContainer<T>&  oc)
{
    options.push_back(oc);
    oc.parse(args);
    return *this;
}

inline
bool
Options::operator>>(const PresentContainer& pc)
{
    options.push_back(pc);

    for(std::list<std::string>::iterator cur = args.begin(); cur != args.end(); ++cur)
    {
        std::string short_opt = "-"; short_opt += pc.s;
        std::string long_opt  = "--" + pc.l;
        if (*cur == short_opt || *cur == long_opt)
        {
            args.erase(cur);
            return true;
        }
    }
    return false;
}

template<class T>
Options&
Options::operator>>(const PosOptionContainer<T>& poc)
{
    failed = !poc.parse(args);
    return *this;
}

}

#endif
